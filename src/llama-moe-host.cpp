#include "llama-moe-host.h"

#include "llama-impl.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define LLAMA_MOE_HOST_PAUSE() _mm_pause()
#else
#define LLAMA_MOE_HOST_PAUSE() std::this_thread::yield()
#endif

namespace {

int64_t moe_host_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// spin barrier of a fixed number of threads
struct moe_host_barrier {
    int n = 0;
    std::atomic<int>      count{0};
    std::atomic<uint32_t> phase{0};

    void wait() {
        const uint32_t ph = phase.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            phase.fetch_add(1, std::memory_order_release);
        } else {
            while (phase.load(std::memory_order_acquire) == ph) {
                LLAMA_MOE_HOST_PAUSE();
            }
        }
    }
};

// gate, up or down of a layer
struct moe_host_mat {
    const char *      data     = nullptr;
    size_t            nb1      = 0; // bytes per row
    size_t            nb2      = 0; // bytes per expert
    ggml_vec_dot_t    dot      = nullptr;
    ggml_type         vdt      = GGML_TYPE_COUNT;
    ggml_from_float_t quantize = nullptr; // to vdt
    size_t            qrow     = 0;       // bytes of an input row quantized to vdt
};

struct moe_host_layer {
    bool         on = false;
    moe_host_mat m[3]; // gate, up, down
    int          x_slot[2] = { 0, 0 }; // the quantized input read by gate and by up
    int          n_x   = 1;            // distinct quantized inputs
    float        limit = 0.0f;
};

bool moe_host_mat_init(moe_host_mat & m, const ggml_tensor * t, int64_t n_in, int64_t n_out) {
    if (t == nullptr || t->data == nullptr || t->ne[0] != n_in || t->ne[1] != n_out || t->nb[0] != ggml_type_size(t->type)) {
        return false;
    }
    const auto * tr = ggml_get_type_traits_cpu(t->type);
    if (tr->vec_dot == nullptr || tr->vec_dot_type == GGML_TYPE_F32 || tr->vec_dot_type == t->type) {
        return false;
    }
    const auto * tq = ggml_get_type_traits_cpu(tr->vec_dot_type);
    if (tq->from_float == nullptr || n_in % ggml_blck_size(t->type) != 0 || n_in % ggml_blck_size(tr->vec_dot_type) != 0) {
        return false;
    }
    m.data     = (const char *) t->data;
    m.nb1      = t->nb[1];
    m.nb2      = t->nb[2];
    m.dot      = tr->vec_dot;
    m.vdt      = tr->vec_dot_type;
    m.quantize = tq->from_float;
    m.qrow     = ggml_row_size(tr->vec_dot_type, n_in);
    return true;
}

}

llama_moe_host::mailbox_layout llama_moe_host::layout(int32_t max_tokens, int32_t n_used, int32_t n_embd, int32_t n_layer, int32_t n_expert) {
    mailbox_layout res;
    size_t off = GGML_PAD(GGML_MOE_HOST_HDR_WORDS*sizeof(int32_t), 128);
    auto take = [&](size_t size) {
        const size_t pos = off;
        off = GGML_PAD(off + size, 128);
        return (int32_t) pos;
    };
    const size_t n_ids = (size_t) max_tokens*n_used;
    res.off_ids   = take(n_ids*sizeof(int32_t));
    res.off_slots = take(n_ids*sizeof(int32_t));
    res.off_w     = take(n_ids*sizeof(float));
    res.off_x     = take((size_t) max_tokens*n_embd*sizeof(float));
    res.off_y     = take((size_t) max_tokens*n_embd*sizeof(float));
    res.off_table = take((size_t) n_layer*n_expert*sizeof(int32_t));
    res.size      = off;
    return res;
}

bool llama_moe_host::supported(const layer & l, int32_t n_embd, int32_t n_ff) {
    moe_host_mat m;
    return moe_host_mat_init(m, l.gate, n_embd, n_ff) &&
           moe_host_mat_init(m, l.up,   n_embd, n_ff) &&
           moe_host_mat_init(m, l.down, n_ff,   n_embd);
}

struct llama_moe_host::impl {
    int32_t n_layer;
    int32_t n_expert;
    int32_t n_used;
    int32_t n_embd;
    int32_t n_ff;
    int32_t max_tokens;
    int32_t skip_rank;
    float   skip_share = 0.0f; // LLAMA_MOE_CACHE_TAIL_SKIP_W: a miss below this share of every token's weight is skipped
    std::vector<float> tok_wsum;

    mailbox_layout lay;
    std::vector<moe_host_layer> layers;

    // a request: its missing experts, each with its entries (token and weight), grouped by expert
    struct cpu_expert {
        int32_t e;
        int32_t first;
        int32_t count;
        int32_t min_rank;
        float   max_share; // the largest share of a token's routing weight it gets
    };
    std::vector<cpu_expert> experts;
    std::vector<int32_t>    ent_tok;
    std::vector<float>      ent_w;
    std::vector<int32_t>    tok_list; // tokens with at least one entry
    std::vector<uint8_t>    tok_has;
    std::vector<int32_t>    stamp;    // [n_expert] request of the last sighting
    std::vector<int32_t>    index;    // [n_expert] position in experts, -1 if skipped
    int32_t                 stamp_gen = 0;

    // scratch of a request
    size_t               qx_row = 0;  // the largest quantized input row
    size_t               qh_row = 0;  // the largest quantized activation row
    std::vector<uint8_t> qx[2];       // [max_tokens][qx_row]
    std::vector<float>   g;           // [max_entries][n_ff], the activation once computed
    std::vector<float>   u;           // [max_entries][n_ff]
    std::vector<uint8_t> qh;          // [max_entries][qh_row]
    std::vector<std::vector<float>> acc; // per thread [max_tokens]

    // the request being computed
    const moe_host_layer * cur   = nullptr;
    const float *          cur_x = nullptr;
    float *                cur_y = nullptr;
    int32_t                cur_n_tokens = 0;

    // phases: quantize the inputs, rows of gate and up, activation, rows of down
    static constexpr int n_phases = 4;
    static constexpr int32_t chunk_a = 32; // rows of gate or up per item
    static constexpr int32_t chunk_d = 64; // rows of down per item
    int32_t n_items[n_phases] = {};
    std::atomic<int32_t> next[n_phases];
    int32_t n_chunks_a = 0;

    // pool: the service thread is thread 0
    int n_workers = 0;
    std::vector<std::thread> workers;
    std::atomic<uint32_t> job_gen{0};
    std::atomic<int>      job_left{0};
    std::atomic<int>      n_sleeping{0};
    std::atomic<int64_t>  active_until{0};
    std::mutex              mtx_pool;
    std::condition_variable cv_pool;
    moe_host_barrier        bar;

    // service
    static constexpr int max_mailboxes = 16;
    std::thread                  service;
    char *                       mailboxes[max_mailboxes] = {};
    int32_t                      last_seq[max_mailboxes]  = {};
    std::atomic<int>             n_mailboxes{0};
    std::atomic<bool>            quit{false};
    std::atomic<int64_t>         wake_until{0};
    std::mutex                   mtx_wake;
    std::condition_variable      cv_wake;
    bool                         woken = false;

    // selection counts since the last take_counts, [n_layer*n_expert]
    std::vector<std::atomic<uint32_t>> counts;

    listener * lis = nullptr;

    // LLAMA_MOE_HOST_TRACE: per request, i32 layer, tokens, experts per token, then the selected experts and their slots
    FILE * trace = nullptr;

    // stats, written by the service thread
    uint64_t st_req      = 0;
    uint64_t st_req_cpu  = 0;
    uint64_t st_tokens   = 0;
    uint64_t st_ids      = 0;
    uint64_t st_misses   = 0; // entries not cached
    uint64_t st_experts  = 0; // experts computed
    uint64_t st_entries  = 0; // entries computed
    uint64_t st_skipped  = 0; // entries skipped
    uint64_t st_ns       = 0;
    uint64_t st_max_ns   = 0;
    uint64_t st_serve_ns = 0; // from seeing a request to answering it
    uint64_t st_turn_ns  = 0; // from an answer to the next request of the same graph: the device's own work
    uint64_t st_turn_n   = 0;
    int64_t  t_answer    = 0;
    bool     warned      = false;

    // LLAMA_MOE_HOST_STATS=<n>: the misses of every n requests, to follow the hit rate along a conversation
    uint64_t st_every    = 0;
    uint64_t st_prev[5]  = {}; // requests, tokens, entries, misses, skipped at the last report

    // a missing expert is computed while the device answer waits, so the pool spins between the requests of a graph
    static constexpr int64_t spin_ns = 50*1000*1000;

    impl(std::vector<layer> ls, int32_t n_expert, int32_t n_used, int32_t n_embd, int32_t n_ff, int32_t max_tokens, int n_threads, int32_t skip_rank) :
            n_layer((int32_t) ls.size()), n_expert(n_expert), n_used(n_used), n_embd(n_embd), n_ff(n_ff),
            max_tokens(max_tokens), skip_rank(skip_rank), counts((size_t) ls.size()*n_expert) {
        lay = layout(max_tokens, n_used, n_embd, (int32_t) ls.size(), n_expert);
        layers.resize(ls.size());
        for (size_t il = 0; il < ls.size(); ++il) {
            moe_host_layer & l = layers[il];
            if (ls[il].gate == nullptr) {
                continue;
            }
            l.on = moe_host_mat_init(l.m[0], ls[il].gate, n_embd, n_ff) &&
                   moe_host_mat_init(l.m[1], ls[il].up,   n_embd, n_ff) &&
                   moe_host_mat_init(l.m[2], ls[il].down, n_ff,   n_embd);
            if (!l.on) {
                continue;
            }
            l.limit     = ls[il].limit;
            l.x_slot[0] = 0;
            l.x_slot[1] = l.m[1].vdt == l.m[0].vdt ? 0 : 1;
            l.n_x       = l.x_slot[1] + 1;
            qx_row = std::max({ qx_row, l.m[0].qrow, l.m[1].qrow });
            qh_row = std::max(qh_row, l.m[2].qrow);
        }

        const size_t max_entries = (size_t) max_tokens*n_used;
        for (auto & q : qx) {
            q.resize((size_t) max_tokens*qx_row);
        }
        g.resize(max_entries*n_ff);
        u.resize(max_entries*n_ff);
        qh.resize(max_entries*qh_row);
        ent_tok.resize(max_entries);
        ent_w.resize(max_entries);
        tok_has.resize(max_tokens);
        stamp.assign(n_expert, 0);
        index.assign(n_expert, -1);
        experts.reserve(max_entries);

        if (const char * v = getenv("LLAMA_MOE_CACHE_TAIL_SKIP_W"); v != nullptr && *v != '\0') {
            skip_share = std::max(0.0f, (float) atof(v));
        }
        tok_wsum.resize(max_tokens);
        if (skip_rank > 0 || skip_share > 0.0f) {
            LLAMA_LOG_INFO("llama_moe_host: tail skip: a missing expert that every token ranks %d-th or later and gives less than %.1f%% of its routing weight is skipped\n",
                skip_rank > 0 ? skip_rank + 1 : 1, skip_share > 0.0f ? 100.0*skip_share : 100.0);
        }
        if (const char * path = getenv("LLAMA_MOE_HOST_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
        }
        if (const char * v = getenv("LLAMA_MOE_HOST_STATS"); v != nullptr && *v != '\0') {
            st_every = strtoull(v, nullptr, 10);
        }

        n_workers = std::max(0, n_threads - 1);
        bar.n = n_workers + 1;
        acc.resize(n_workers + 1, std::vector<float>(max_tokens));
        for (auto & n : next) {
            n.store(0);
        }
        for (int i = 0; i < n_workers; ++i) {
            workers.emplace_back([this, i] { worker_loop(i + 1); });
        }
        service = std::thread([this] { service_loop(); });
    }

    ~impl() {
        quit.store(true);
        {
            std::lock_guard<std::mutex> lk(mtx_wake);
            woken = true;
        }
        cv_wake.notify_all();
        {
            std::lock_guard<std::mutex> lk(mtx_pool);
        }
        cv_pool.notify_all();
        if (service.joinable()) {
            service.join();
        }
        for (auto & t : workers) {
            t.join();
        }
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    void worker_loop(int ith) {
        uint32_t seen = 0;
        for (;;) {
            uint32_t g;
            int spins = 0;
            while ((g = job_gen.load(std::memory_order_acquire)) == seen) {
                if (quit.load(std::memory_order_relaxed)) {
                    return;
                }
                if (++spins >= 1024) {
                    spins = 0;
                    if (moe_host_now_ns() > active_until.load(std::memory_order_relaxed)) {
                        std::unique_lock<std::mutex> lk(mtx_pool);
                        n_sleeping.fetch_add(1);
                        cv_pool.wait(lk, [&] {
                            return job_gen.load() != seen || quit.load() || moe_host_now_ns() <= active_until.load();
                        });
                        n_sleeping.fetch_sub(1);
                    }
                }
                LLAMA_MOE_HOST_PAUSE();
            }
            seen = g;
            run_job(ith);
            job_left.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    void keep_spinning(int64_t until) {
        if (until > active_until.load(std::memory_order_relaxed)) {
            active_until.store(until, std::memory_order_relaxed);
        }
        if (n_sleeping.load() > 0) {
            {
                std::lock_guard<std::mutex> lk(mtx_pool);
            }
            cv_pool.notify_all();
        }
    }

    void run_all() {
        job_left.store(n_workers, std::memory_order_relaxed);
        job_gen.fetch_add(1);
        keep_spinning(moe_host_now_ns() + spin_ns);
        run_job(0);
        while (job_left.load(std::memory_order_acquire) > 0) {
            LLAMA_MOE_HOST_PAUSE();
        }
    }

    static float silu(float x) {
        return x/(1.0f + expf(-x));
    }

    void run_job(int ith) {
        const moe_host_layer & l = *cur;
        const int n_x = l.n_x;

        // the inputs of the tokens with work, quantized for gate and up
        for (int32_t i; (i = next[0].fetch_add(1, std::memory_order_relaxed)) < n_items[0]; ) {
            const int     xs = i % n_x;
            const int32_t t  = tok_list[i / n_x];
            const moe_host_mat & m = l.m[xs == l.x_slot[0] ? 0 : 1];
            m.quantize(cur_x + (size_t) t*n_embd, qx[xs].data() + (size_t) t*qx_row, n_embd);
        }
        bar.wait();

        // rows of gate and up, for every token of an expert
        for (int32_t i; (i = next[1].fetch_add(1, std::memory_order_relaxed)) < n_items[1]; ) {
            const int32_t c  = i % n_chunks_a;
            const int     mi = (i / n_chunks_a) % 2;
            const cpu_expert & ce = experts[i / (2*n_chunks_a)];
            const moe_host_mat & m = l.m[mi];
            const uint8_t * qxs = qx[l.x_slot[mi]].data();
            float * out = mi == 0 ? g.data() : u.data();
            const int32_t r0 = c*chunk_a;
            const int32_t r1 = std::min(r0 + chunk_a, n_ff);
            const char * w = m.data + (size_t) ce.e*m.nb2;
            for (int32_t r = r0; r < r1; ++r) {
                const char * wr = w + (size_t) r*m.nb1;
                for (int32_t q = ce.first; q < ce.first + ce.count; ++q) {
                    float v;
                    m.dot(n_embd, &v, 0, wr, 0, qxs + (size_t) ent_tok[q]*qx_row, 0, 1);
                    out[(size_t) q*n_ff + r] = v;
                }
            }
        }
        bar.wait();

        // the activation of each entry, quantized for down
        for (int32_t q; (q = next[2].fetch_add(1, std::memory_order_relaxed)) < n_items[2]; ) {
            float       * gq = g.data() + (size_t) q*n_ff;
            const float * uq = u.data() + (size_t) q*n_ff;
            if (l.limit > 0.0f) {
                for (int32_t i = 0; i < n_ff; ++i) {
                    gq[i] = std::min(silu(gq[i]), l.limit)*std::clamp(uq[i], -l.limit, l.limit);
                }
            } else {
                for (int32_t i = 0; i < n_ff; ++i) {
                    gq[i] = silu(gq[i])*uq[i];
                }
            }
            l.m[2].quantize(gq, qh.data() + (size_t) q*qh_row, n_ff);
        }
        bar.wait();

        // rows of down, summed over the experts of each token with their weights
        float * a = acc[ith].data();
        const moe_host_mat & md = l.m[2];
        for (int32_t i; (i = next[3].fetch_add(1, std::memory_order_relaxed)) < n_items[3]; ) {
            const int32_t r0 = i*chunk_d;
            const int32_t r1 = std::min(r0 + chunk_d, n_embd);
            for (int32_t r = r0; r < r1; ++r) {
                std::fill(a, a + cur_n_tokens, 0.0f);
                for (const cpu_expert & ce : experts) {
                    const char * wr = md.data + (size_t) ce.e*md.nb2 + (size_t) r*md.nb1;
                    for (int32_t q = ce.first; q < ce.first + ce.count; ++q) {
                        float v;
                        md.dot(n_ff, &v, 0, wr, 0, qh.data() + (size_t) q*qh_row, 0, 1);
                        a[ent_tok[q]] += ent_w[q]*v;
                    }
                }
                for (int32_t t = 0; t < cur_n_tokens; ++t) {
                    cur_y[(size_t) t*n_embd + r] = a[t];
                }
            }
        }
    }

    // returns the layer of the request, -1 if it is invalid
    int32_t serve(char * mb, int32_t seq) {
        int32_t * hdr = (int32_t *) mb;
        const int32_t il = hdr[GGML_MOE_HOST_REQ_LAYER];
        auto answer = [&](int32_t has_y) {
            if (lis != nullptr && il >= 0 && il < n_layer && layers[il].on) {
                lis->before_answer(il);
            }
            hdr[GGML_MOE_HOST_RESP_HAS_Y] = has_y;
            __atomic_store_n(hdr + GGML_MOE_HOST_RESP_SEQ, seq, __ATOMIC_RELEASE);
        };

        const int32_t n_tokens = hdr[GGML_MOE_HOST_REQ_N_TOKENS];
        if (il < 0 || il >= n_layer || !layers[il].on || n_tokens <= 0 || n_tokens > max_tokens || hdr[GGML_MOE_HOST_REQ_N_USED] != n_used) {
            if (!warned) {
                LLAMA_LOG_ERROR("llama_moe_host: invalid request: layer %d, %d tokens, %d experts per token\n", il, n_tokens, hdr[GGML_MOE_HOST_REQ_N_USED]);
                warned = true;
            }
            answer(0);
            return -1;
        }

        const int32_t * ids   = (const int32_t *) (mb + lay.off_ids);
        const int32_t * slots = (const int32_t *) (mb + lay.off_slots);
        const float   * w     = (const float   *) (mb + lay.off_w);
        const int32_t   n     = n_tokens*n_used;

        st_req++;
        st_tokens += n_tokens;
        st_ids    += n;

        if (trace != nullptr) {
            const int32_t rec[3] = { il, n_tokens, n_used };
            fwrite(rec, sizeof(int32_t), 3, trace);
            fwrite(ids, sizeof(int32_t), n, trace);
            fwrite(slots, sizeof(int32_t), n, trace);
        }

        std::atomic<uint32_t> * cnt = counts.data() + (size_t) il*n_expert;
        for (int32_t i = 0; i < n; ++i) {
            std::atomic<uint32_t> & c = cnt[ids[i]];
            c.store(c.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        }

        for (int32_t t = 0; t < n_tokens; ++t) {
            float sum = 0.0f;
            for (int32_t k = 0; k < n_used; ++k) {
                sum += w[t*n_used + k];
            }
            tok_wsum[t] = sum > 0.0f ? sum : 1.0f;
        }

        // the missing experts, the lowest rank and the largest weight share each one has in the tokens that select it
        if (++stamp_gen == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            stamp_gen = 1;
        }
        experts.clear();
        int32_t n_miss = 0;
        for (int32_t i = 0; i < n; ++i) {
            if (slots[i] >= 0) {
                continue;
            }
            n_miss++;
            const int32_t e = ids[i];
            const int32_t k = i % n_used;
            if (stamp[e] != stamp_gen) {
                stamp[e] = stamp_gen;
                index[e] = (int32_t) experts.size();
                experts.push_back({ e, 0, 0, k, 0.0f });
            }
            cpu_expert & ce = experts[index[e]];
            ce.count++;
            ce.min_rank  = std::min(ce.min_rank, k);
            ce.max_share = std::max(ce.max_share, w[i]/tok_wsum[i / n_used]);
        }
        st_misses += n_miss;

        // the tail skip, then the entries of the computed experts grouped by expert
        int32_t n_ent = 0;
        size_t kept = 0;
        for (size_t j = 0; j < experts.size(); ++j) {
            cpu_expert ce = experts[j];
            const bool skip = (skip_rank > 0 || skip_share > 0.0f) &&
                (skip_rank   <= 0    || ce.min_rank  >= skip_rank) &&
                (skip_share  <= 0.0f || ce.max_share <  skip_share);
            if (skip) {
                index[ce.e] = -1;
                st_skipped += ce.count;
                continue;
            }
            index[ce.e] = (int32_t) kept;
            ce.first = n_ent;
            n_ent   += ce.count;
            ce.count = 0;
            experts[kept++] = ce;
        }
        experts.resize(kept);
        if (experts.empty()) {
            answer(0);
            return il;
        }

        std::fill(tok_has.begin(), tok_has.begin() + n_tokens, 0);
        for (int32_t i = 0; i < n; ++i) {
            if (slots[i] >= 0 || index[ids[i]] < 0) {
                continue;
            }
            cpu_expert & ce = experts[index[ids[i]]];
            const int32_t q = ce.first + ce.count++;
            ent_tok[q] = i / n_used;
            ent_w[q]   = w[i];
            tok_has[i / n_used] = 1;
        }
        tok_list.clear();
        for (int32_t t = 0; t < n_tokens; ++t) {
            if (tok_has[t]) {
                tok_list.push_back(t);
            }
        }

        cur          = &layers[il];
        cur_x        = (const float *) (mb + lay.off_x);
        cur_y        = (float *) (mb + lay.off_y);
        cur_n_tokens = n_tokens;
        n_chunks_a   = (n_ff + chunk_a - 1)/chunk_a;
        n_items[0]   = (int32_t) tok_list.size()*cur->n_x;
        n_items[1]   = (int32_t) experts.size()*2*n_chunks_a;
        n_items[2]   = n_ent;
        n_items[3]   = (n_embd + chunk_d - 1)/chunk_d;
        for (auto & nx : next) {
            nx.store(0, std::memory_order_relaxed);
        }

        const int64_t t0 = moe_host_now_ns();
        run_all();
        const uint64_t dt = moe_host_now_ns() - t0;

        st_req_cpu++;
        st_experts += experts.size();
        st_entries += n_ent;
        st_ns      += dt;
        st_max_ns   = std::max(st_max_ns, dt);

        answer(1);
        return il;
    }

    void service_loop() {
        int64_t last_req = moe_host_now_ns();
        for (;;) {
            bool got = false;
            const int n_mb = n_mailboxes.load(std::memory_order_acquire);
            for (int m = 0; m < n_mb; ++m) {
                int32_t * hdr = (int32_t *) mailboxes[m];
                const int32_t seq = __atomic_load_n(hdr + GGML_MOE_HOST_REQ_SEQ, __ATOMIC_ACQUIRE);
                if (seq != last_seq[m]) {
                    last_seq[m] = seq;
                    const int64_t t_seen = moe_host_now_ns();
                    if (t_seen - t_answer < 5*1000*1000) {
                        st_turn_ns += t_seen - t_answer;
                        st_turn_n++;
                    }
                    const int32_t il = serve(mailboxes[m], seq);
                    t_answer = moe_host_now_ns();
                    st_serve_ns += t_answer - t_seen;
                    if (lis != nullptr && il >= 0) {
                        lis->after_answer(il);
                    }
                    if (st_every > 0 && st_req - st_prev[0] >= st_every) {
                        const uint64_t ids = st_ids - st_prev[2];
                        LLAMA_LOG_INFO("llama_moe_host: last %llu requests (%.1f tokens): %.2f%% of the selected experts missing, %.2f%% skipped\n",
                            (unsigned long long) (st_req - st_prev[0]), (double) (st_tokens - st_prev[1])/(st_req - st_prev[0]),
                            100.0*(st_misses - st_prev[3])/ids, 100.0*(st_skipped - st_prev[4])/ids);
                        const uint64_t cur[5] = { st_req, st_tokens, st_ids, st_misses, st_skipped };
                        std::copy(cur, cur + 5, st_prev);
                    }
                    got = true;
                }
            }
            if (got) {
                last_req = moe_host_now_ns();
                continue;
            }
            if (quit.load(std::memory_order_relaxed)) {
                return;
            }
            const int64_t now = moe_host_now_ns();
            if (now - last_req > spin_ns && now > wake_until.load(std::memory_order_relaxed)) {
                std::unique_lock<std::mutex> lk(mtx_wake);
                cv_wake.wait_for(lk, std::chrono::microseconds(200), [&] { return woken; });
                if (woken) {
                    woken    = false;
                    last_req = moe_host_now_ns();
                }
            } else {
                LLAMA_MOE_HOST_PAUSE();
            }
        }
    }

    void wake() {
        const int64_t until = moe_host_now_ns() + spin_ns;
        wake_until.store(until, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mtx_wake);
            woken = true;
        }
        cv_wake.notify_one();
        keep_spinning(until);
    }
};

llama_moe_host::llama_moe_host(std::vector<layer> layers, int32_t n_expert, int32_t n_used, int32_t n_embd, int32_t n_ff,
        int32_t max_tokens, int n_threads, int32_t skip_rank) :
    pimpl(new impl(std::move(layers), n_expert, n_used, n_embd, n_ff, max_tokens, n_threads, skip_rank)) {
}

llama_moe_host::~llama_moe_host() = default;

void llama_moe_host::add_mailbox(void * mailbox) {
    const int n = pimpl->n_mailboxes.load();
    GGML_ASSERT(n < impl::max_mailboxes);
    pimpl->mailboxes[n] = (char *) mailbox;
    pimpl->last_seq[n]  = ((const int32_t *) mailbox)[GGML_MOE_HOST_REQ_SEQ];
    pimpl->n_mailboxes.store(n + 1, std::memory_order_release);
}

void llama_moe_host::wake() {
    pimpl->wake();
}

void llama_moe_host::set_listener(listener * l) {
    pimpl->lis = l;
}

void llama_moe_host::take_counts(std::vector<float> & counts, float decay, std::vector<float> * total) {
    GGML_ASSERT(counts.size() == pimpl->counts.size());
    GGML_ASSERT(total == nullptr || total->size() == counts.size());
    for (size_t i = 0; i < counts.size(); ++i) {
        const float c = (float) pimpl->counts[i].exchange(0, std::memory_order_relaxed);
        counts[i] = counts[i]*decay + c;
        if (total != nullptr) {
            (*total)[i] += c;
        }
    }
}

void llama_moe_host::log_stats() const {
    const impl & p = *pimpl;
    if (p.st_req == 0) {
        return;
    }
    LLAMA_LOG_INFO("llama_moe_host: %llu requests (%.1f tokens), %.2f%% of the selected experts missing, %.2f%% computed on the CPU, %.2f%% skipped\n",
        (unsigned long long) p.st_req, (double) p.st_tokens/p.st_req, 100.0*p.st_misses/p.st_ids,
        100.0*p.st_entries/p.st_ids, 100.0*p.st_skipped/p.st_ids);
    LLAMA_LOG_INFO("llama_moe_host: %.1f us from a request to its answer on average, %.1f us from an answer to the next request\n",
        p.st_serve_ns/1e3/p.st_req, p.st_turn_n > 0 ? p.st_turn_ns/1e3/p.st_turn_n : 0.0);
    if (p.st_req_cpu > 0) {
        LLAMA_LOG_INFO("llama_moe_host: %llu requests with CPU work: %.2f experts, %.2f entries, %.1f us each on average, %.1f us at most\n",
            (unsigned long long) p.st_req_cpu, (double) p.st_experts/p.st_req_cpu, (double) p.st_entries/p.st_req_cpu,
            p.st_ns/1e3/p.st_req_cpu, p.st_max_ns/1e3);
    }
}
