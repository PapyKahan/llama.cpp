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
    ggml_type         type     = GGML_TYPE_COUNT;
    int               row8_min = 0;       // an expert with this many entries or more unpacks each row once, 0 = never
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
    m.type     = t->type;
    // an unpacked row costs about one dot product and the next ones are much cheaper; the IQ4 and Q2_0 dot products
    // are cheap already, so the unpacking pays from a third entry
    if (ggml_cpu_row8_supported(t->type) && n_in % 32 == 0) {
        const bool cheap = t->type == GGML_TYPE_IQ4_XS || t->type == GGML_TYPE_IQ4_NL || t->type == GGML_TYPE_Q2_0;
        m.row8_min = cheap ? 3 : 2;
        if (const char * v = getenv("LLAMA_MOE_HOST_ROW8"); v != nullptr && *v != '\0') {
            m.row8_min = atoi(v) > 0 ? std::max(m.row8_min, atoi(v)) : 0;
        }
    }
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
    res.off_yh    = take((size_t) max_tokens*n_embd*sizeof(ggml_fp16_t));
    res.off_yflag = take((size_t) max_tokens*sizeof(int32_t));
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

    // per thread, a row unpacked for the entries of an expert (ggml_cpu_row8_unpack) and their activation rows
    struct row8_buf {
        std::vector<int8_t>       q;
        std::vector<int16_t>      sc;
        std::vector<float>        d;
        std::vector<const void *> ys;
        std::vector<float>        out;
    };
    std::vector<row8_buf> r8;

    // whole large batches (add_batch_mailbox): the entries of each expert the host computes, its busiest experts first,
    // each worker taking an expert and passing over its rows once per batch_chunk entries; a token's row of y is
    // written by the first expert that reaches it and added to by the others, under the token's lock
    static constexpr int32_t batch_chunk = 64;
    int32_t                  max_batch = 0;
    std::vector<int32_t>     b_cnt;     // [n_expert]
    std::vector<int32_t>     b_first;   // [n_expert]
    std::vector<int32_t>     b_experts;
    std::vector<int32_t>     b_items;   // expert*n_chunks_max + chunk, the busiest experts' chunks first
    std::vector<int32_t>     b_tok;     // [entries]
    std::vector<float>       b_w;       // [entries]
    std::vector<uint8_t>     b_has;     // [max_batch]
    std::vector<int32_t>     b_toks;    // tokens with entries
    std::vector<uint8_t>     b_qx[2];   // [max_batch][qx_row]
    std::unique_ptr<std::atomic<uint8_t>[]> b_lock; // [max_batch]
    std::vector<uint8_t>     b_touched; // [max_batch]
    struct batch_buf {
        std::vector<float>   g;  // [batch_chunk][n_ff]
        std::vector<float>   u;  // [batch_chunk][n_ff]
        std::vector<uint8_t> qa; // [batch_chunk][qh_row]
        std::vector<float>   o;  // [batch_chunk][n_embd]
    };
    std::vector<batch_buf>   bb;        // per thread
    std::atomic<int32_t>     b_next[3];
    ggml_from_float_t        q80_from_float = nullptr;
    ggml_fp16_t *            cur_yh = nullptr;
    bool                     cur_batch = false;
    uint64_t st_breq = 0, st_bexperts = 0, st_bentries = 0, st_bns = 0;

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
    bool                         is_batch[max_mailboxes]  = {};
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
        r8.resize(n_workers + 1);
        for (row8_buf & b : r8) {
            const size_t n = (size_t) std::max(n_embd, n_ff);
            b.q.resize(n);
            b.sc.resize(n/16);
            b.d.resize(n/32);
            b.ys.resize(std::max(max_tokens, batch_chunk));
            b.out.resize(std::max(max_tokens, batch_chunk));
        }
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
        if (cur_batch) {
            run_job_batch(ith);
            return;
        }
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
            if (m.row8_min > 0 && ce.count >= m.row8_min) {
                row8_buf & b = r8[ith];
                for (int32_t j = 0; j < ce.count; ++j) {
                    b.ys[j] = qxs + (size_t) ent_tok[ce.first + j]*qx_row;
                }
                for (int32_t r = r0; r < r1; ++r) {
                    ggml_cpu_row8_unpack(m.type, w + (size_t) r*m.nb1, n_embd, b.q.data(), b.sc.data(), b.d.data());
                    ggml_cpu_row8_dot(m.vdt, b.q.data(), b.sc.data(), b.d.data(), b.ys.data(), ce.count, n_embd, b.out.data());
                    for (int32_t j = 0; j < ce.count; ++j) {
                        out[(size_t) (ce.first + j)*n_ff + r] = b.out[j];
                    }
                }
                continue;
            }
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
                    if (md.row8_min > 0 && ce.count >= md.row8_min) {
                        row8_buf & b = r8[ith];
                        for (int32_t j = 0; j < ce.count; ++j) {
                            b.ys[j] = qh.data() + (size_t) (ce.first + j)*qh_row;
                        }
                        ggml_cpu_row8_unpack(md.type, wr, n_ff, b.q.data(), b.sc.data(), b.d.data());
                        ggml_cpu_row8_dot(md.vdt, b.q.data(), b.sc.data(), b.d.data(), b.ys.data(), ce.count, n_ff, b.out.data());
                        for (int32_t j = 0; j < ce.count; ++j) {
                            a[ent_tok[ce.first + j]] += ent_w[ce.first + j]*b.out[j];
                        }
                        continue;
                    }
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

    // one row of m for the entries [q0, q0 + k), whose inputs are ys; out[j] for entry q0 + j
    // q80: the inputs are Q8_0 whatever the type, for its unpacked rows
    static void batch_rows(const moe_host_mat & m, const char * w, int32_t n_in, int32_t r0, int32_t r1, const void ** ys,
            int32_t k, row8_buf & b, float * dst, int32_t dst_stride, bool q80 = false) {
        const bool unpack = q80 || (m.row8_min > 0 && k >= m.row8_min);
        for (int32_t r = r0; r < r1; ++r) {
            const char * wr = w + (size_t) r*m.nb1;
            if (unpack) {
                ggml_cpu_row8_unpack(m.type, wr, n_in, b.q.data(), b.sc.data(), b.d.data());
                ggml_cpu_row8_dot(q80 ? GGML_TYPE_Q8_0 : m.vdt, b.q.data(), b.sc.data(), b.d.data(), ys, k, n_in, b.out.data());
                for (int32_t j = 0; j < k; ++j) {
                    dst[(size_t) j*dst_stride + r] = b.out[j];
                }
            } else {
                for (int32_t j = 0; j < k; ++j) {
                    float v;
                    m.dot(n_in, &v, 0, wr, 0, ys[j], 0, 1);
                    dst[(size_t) j*dst_stride + r] = v;
                }
            }
        }
    }

    void run_job_batch(int ith) {
        const moe_host_layer & l = *cur;
        const int n_x = l.n_x;

        // the inputs of the tokens with entries, quantized for gate and up: to Q8_0 when both unpack their rows, whose
        // scale per 32 inputs is closer to the device's activations than the vec_dot types' per 256
        const bool    q80 = l.m[0].row8_min > 0 && l.m[1].row8_min > 0;
        const int     n_xq = q80 ? 1 : n_x;
        const int32_t n_q  = (int32_t) b_toks.size()*n_xq;
        for (int32_t i; (i = b_next[0].fetch_add(1, std::memory_order_relaxed)) < n_q; ) {
            const int     xs = i % n_xq;
            const int32_t t  = b_toks[i / n_xq];
            if (q80) {
                q80_from_float(cur_x + (size_t) t*n_embd, b_qx[0].data() + (size_t) t*qx_row, n_embd);
                continue;
            }
            const moe_host_mat & m = l.m[xs == l.x_slot[0] ? 0 : 1];
            m.quantize(cur_x + (size_t) t*n_embd, b_qx[xs].data() + (size_t) t*qx_row, n_embd);
        }
        bar.wait();

        row8_buf  & b  = r8[ith];
        batch_buf & bf = bb[ith];
        const void * ys[batch_chunk];
        const int32_t n_chunks_max = (max_batch + batch_chunk - 1)/batch_chunk;
        for (int32_t i; (i = b_next[1].fetch_add(1, std::memory_order_relaxed)) < (int32_t) b_items.size(); ) {
            const int32_t e     = b_items[i] / n_chunks_max;
            const int32_t c0    = (b_items[i] % n_chunks_max)*batch_chunk;
            const int32_t count = b_cnt[e];
            {
                const int32_t k  = std::min(batch_chunk, count - c0);
                const int32_t q0 = b_first[e] + c0;

                // gate and up
                for (int mi = 0; mi < 2; ++mi) {
                    const moe_host_mat & m = l.m[mi];
                    const uint8_t * qxs = b_qx[q80 ? 0 : l.x_slot[mi]].data();
                    for (int32_t j = 0; j < k; ++j) {
                        ys[j] = qxs + (size_t) b_tok[q0 + j]*qx_row;
                    }
                    batch_rows(m, m.data + (size_t) e*m.nb2, n_embd, 0, n_ff, ys, k, b, mi == 0 ? bf.g.data() : bf.u.data(), n_ff, q80);
                }

                // the activation, quantized for down
                const moe_host_mat & md = l.m[2];
                for (int32_t j = 0; j < k; ++j) {
                    float       * gq = bf.g.data() + (size_t) j*n_ff;
                    const float * uq = bf.u.data() + (size_t) j*n_ff;
                    if (l.limit > 0.0f) {
                        for (int32_t r = 0; r < n_ff; ++r) {
                            gq[r] = std::min(silu(gq[r]), l.limit)*std::clamp(uq[r], -l.limit, l.limit);
                        }
                    } else {
                        for (int32_t r = 0; r < n_ff; ++r) {
                            gq[r] = silu(gq[r])*uq[r];
                        }
                    }
                    md.quantize(gq, bf.qa.data() + (size_t) j*qh_row, n_ff);
                    ys[j] = bf.qa.data() + (size_t) j*qh_row;
                }

                // down, then into the rows of y
                batch_rows(md, md.data + (size_t) e*md.nb2, n_ff, 0, n_embd, ys, k, b, bf.o.data(), n_embd);
                for (int32_t j = 0; j < k; ++j) {
                    const int32_t t = b_tok[q0 + j];
                    const float   w = b_w[q0 + j];
                    const float * o = bf.o.data() + (size_t) j*n_embd;
                    float       * y = cur_y + (size_t) t*n_embd;
                    std::atomic<uint8_t> & lk = b_lock[t];
                    while (lk.exchange(1, std::memory_order_acquire)) {
                        LLAMA_MOE_HOST_PAUSE();
                    }
                    if (b_touched[t]) {
                        for (int32_t r = 0; r < n_embd; ++r) {
                            y[r] += w*o[r];
                        }
                    } else {
                        for (int32_t r = 0; r < n_embd; ++r) {
                            y[r] = w*o[r];
                        }
                        b_touched[t] = 1;
                    }
                    lk.store(0, std::memory_order_release);
                }
            }
        }
        bar.wait();

        // the answer of the tokens with entries, in half precision: the device reads only the rows the host flags
        for (int32_t i; (i = b_next[2].fetch_add(1, std::memory_order_relaxed)) < (int32_t) b_toks.size(); ) {
            const int32_t t = b_toks[i];
            ggml_cpu_fp32_to_fp16(cur_y + (size_t) t*n_embd, cur_yh + (size_t) t*n_embd, n_embd);
        }
    }

    // a whole large batch: the entries the device marks -1, grouped by expert; no tail skip, a prompt is computed whole
    int32_t serve_batch(char * mb, int32_t seq) {
        int32_t * hdr = (int32_t *) mb;
        const int32_t il       = hdr[GGML_MOE_HOST_REQ_LAYER];
        const int32_t n_tokens = hdr[GGML_MOE_HOST_REQ_N_TOKENS];
        auto answer = [&](int32_t has_y) {
            hdr[GGML_MOE_HOST_RESP_HAS_Y] = has_y;
            __atomic_store_n(hdr + GGML_MOE_HOST_RESP_SEQ, seq, __ATOMIC_RELEASE);
        };
        if (il < 0 || il >= n_layer || !layers[il].on || n_tokens <= 0 || n_tokens > max_batch || hdr[GGML_MOE_HOST_REQ_N_USED] != n_used) {
            if (!warned) {
                LLAMA_LOG_ERROR("llama_moe_host: invalid batch request: layer %d, %d tokens, %d experts per token\n", il, n_tokens, hdr[GGML_MOE_HOST_REQ_N_USED]);
                warned = true;
            }
            answer(0);
            return -1;
        }
        const mailbox_layout bl = layout(max_batch, n_used, n_embd, n_layer, n_expert);
        const int32_t * ids   = (const int32_t *) (mb + bl.off_ids);
        const int32_t * slots = (const int32_t *) (mb + bl.off_slots);
        const float   * w     = (const float   *) (mb + bl.off_w);
        const int32_t   n     = n_tokens*n_used;

        if (b_lock == nullptr) {
            q80_from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
            GGML_ASSERT(ggml_row_size(GGML_TYPE_Q8_0, n_embd) <= qx_row);
            b_lock.reset(new std::atomic<uint8_t>[max_batch]);
            for (int32_t t = 0; t < max_batch; ++t) {
                b_lock[t].store(0);
            }
            b_touched.resize(max_batch);
            b_has.resize(max_batch);
            b_cnt.resize(n_expert);
            b_first.resize(n_expert);
            b_tok.resize((size_t) max_batch*n_used);
            b_w.resize((size_t) max_batch*n_used);
            for (auto & q : b_qx) {
                q.resize((size_t) max_batch*qx_row);
            }
            bb.resize(n_workers + 1);
            for (batch_buf & bf : bb) {
                bf.g.resize((size_t) batch_chunk*n_ff);
                bf.u.resize((size_t) batch_chunk*n_ff);
                bf.qa.resize((size_t) batch_chunk*qh_row);
                bf.o.resize((size_t) batch_chunk*n_embd);
            }
        }

        // a counting sort of the entries by expert
        std::fill(b_cnt.begin(), b_cnt.end(), 0);
        int32_t n_ent = 0;
        for (int32_t i = 0; i < n; ++i) {
            if (slots[i] < 0) {
                b_cnt[ids[i]]++;
                n_ent++;
            }
        }
        if (n_ent == 0) {
            answer(0);
            return il;
        }
        b_experts.clear();
        for (int32_t e = 0, pos = 0; e < n_expert; ++e) {
            b_first[e] = pos;
            pos += b_cnt[e];
            if (b_cnt[e] > 0) {
                b_experts.push_back(e);
            }
        }
        std::sort(b_experts.begin(), b_experts.end(), [&](int32_t a, int32_t c) { return b_cnt[a] > b_cnt[c]; });
        // a chunk of an expert per work item: a busy expert spreads over several threads, each passing over its rows
        const int32_t n_chunks_max = (max_batch + batch_chunk - 1)/batch_chunk;
        b_items.clear();
        for (int32_t e : b_experts) {
            for (int32_t c = 0; c*batch_chunk < b_cnt[e]; ++c) {
                b_items.push_back(e*n_chunks_max + c);
            }
        }
        std::fill(b_cnt.begin(), b_cnt.end(), 0);
        std::fill_n(b_has.begin(), n_tokens, 0);
        for (int32_t i = 0; i < n; ++i) {
            if (slots[i] < 0) {
                const int32_t e = ids[i];
                const int32_t q = b_first[e] + b_cnt[e]++;
                b_tok[q] = i / n_used;
                b_w[q]   = w[i];
                b_has[i / n_used] = 1;
            }
        }
        b_toks.clear();
        int32_t * yflag = (int32_t *) (mb + bl.off_yflag);
        for (int32_t t = 0; t < n_tokens; ++t) {
            if (b_has[t]) {
                b_toks.push_back(t);
            }
            yflag[t]     = b_has[t];
            b_touched[t] = 0;
        }

        // the host sums the experts of a token in single precision in y, then writes the flagged rows in half
        cur          = &layers[il];
        cur_x        = (const float *) (mb + bl.off_x);
        cur_y        = (float *) (mb + bl.off_y);
        cur_yh       = (ggml_fp16_t *) (mb + bl.off_yh);
        cur_n_tokens = n_tokens;
        cur_batch    = true;
        for (auto & nx : b_next) {
            nx.store(0, std::memory_order_relaxed);
        }

        const int64_t t0 = moe_host_now_ns();
        run_all();
        cur_batch = false;

        st_breq++;
        st_bexperts += b_experts.size();
        st_bentries += n_ent;
        st_bns      += moe_host_now_ns() - t0;

        answer(1);
        return il;
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

        // the entries of a pad token have id -1: no count, no work
        std::atomic<uint32_t> * cnt = counts.data() + (size_t) il*n_expert;
        for (int32_t i = 0; i < n; ++i) {
            if (ids[i] < 0) {
                continue;
            }
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
            if (slots[i] >= 0 || ids[i] < 0) {
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
            if (slots[i] >= 0 || ids[i] < 0 || index[ids[i]] < 0) {
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
                    if (is_batch[m]) {
                        serve_batch(mailboxes[m], seq);
                        t_answer = moe_host_now_ns();
                        got = true;
                        continue;
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

void llama_moe_host::add_batch_mailbox(void * mailbox, int32_t max_batch) {
    const int n = pimpl->n_mailboxes.load();
    GGML_ASSERT(n < impl::max_mailboxes);
    GGML_ASSERT(pimpl->max_batch == 0 || pimpl->max_batch == max_batch);
    pimpl->max_batch    = max_batch;
    pimpl->mailboxes[n] = (char *) mailbox;
    pimpl->is_batch[n]  = true;
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
    if (p.st_breq > 0) {
        LLAMA_LOG_INFO("llama_moe_host: %llu batch requests: %.1f experts, %.1f entries, %.2f ms each on average\n",
            (unsigned long long) p.st_breq, (double) p.st_bexperts/p.st_breq, (double) p.st_bentries/p.st_breq, p.st_bns/1e6/p.st_breq);
    }
}
