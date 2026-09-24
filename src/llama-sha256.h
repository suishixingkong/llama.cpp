#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

inline void llama_store_le_u32(uint8_t (&data)[4], uint32_t value) {
    for (size_t i = 0; i < sizeof(data); ++i) {
        data[i] = uint8_t(value >> (8*i));
    }
}

inline void llama_store_le_u64(uint8_t (&data)[8], uint64_t value) {
    for (size_t i = 0; i < sizeof(data); ++i) {
        data[i] = uint8_t(value >> (8*i));
    }
}

inline uint32_t llama_load_le_u32(const uint8_t (&data)[4]) {
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(data); ++i) {
        value |= uint32_t(data[i]) << (8*i);
    }
    return value;
}

inline uint64_t llama_load_le_u64(const uint8_t (&data)[8]) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(data); ++i) {
        value |= uint64_t(data[i]) << (8*i);
    }
    return value;
}

// Compact SHA-256 for internal identity digests (adapter identities, VBR checkpoint
// identity/policy/order digests). Callers version their own serialization domains.
class llama_sha256 {
public:
    llama_sha256() {
        state = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
        };
    }

    void update(const void * src, size_t len) {
        const uint8_t * data = static_cast<const uint8_t *>(src);
        total_len += len;

        while (len > 0) {
            const size_t n = std::min(len, block.size() - block_len);
            memcpy(block.data() + block_len, data, n);
            block_len += n;
            data += n;
            len -= n;

            if (block_len == block.size()) {
                transform(block.data());
                block_len = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        const uint64_t bit_len = total_len * 8;
        const uint8_t one = 0x80;
        update(&one, 1);

        const uint8_t zero = 0;
        while (block_len != 56) {
            update(&zero, 1);
        }

        uint8_t len_be[8];
        for (size_t i = 0; i < sizeof(len_be); ++i) {
            len_be[i] = uint8_t(bit_len >> (56 - 8*i));
        }
        update(len_be, sizeof(len_be));

        std::array<uint8_t, 32> result;
        for (size_t i = 0; i < state.size(); ++i) {
            result[4*i + 0] = uint8_t(state[i] >> 24);
            result[4*i + 1] = uint8_t(state[i] >> 16);
            result[4*i + 2] = uint8_t(state[i] >>  8);
            result[4*i + 3] = uint8_t(state[i] >>  0);
        }
        return result;
    }

private:
    static uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    void transform(const uint8_t * data) {
        static const uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };

        uint32_t w[64];
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (uint32_t(data[4*i + 0]) << 24) |
                   (uint32_t(data[4*i + 1]) << 16) |
                   (uint32_t(data[4*i + 2]) <<  8) |
                   (uint32_t(data[4*i + 3]) <<  0);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15],  7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>  3);
            const uint32_t s1 = rotr(w[i -  2], 17) ^ rotr(w[i -  2], 19) ^ (w[i -  2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];

        for (size_t i = 0; i < 64; ++i) {
            const uint32_t s1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch  = (e & f) ^ (~e & g);
            const uint32_t t1  = h + s1 + ch + k[i] + w[i];
            const uint32_t s0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2  = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    std::array<uint32_t, 8> state;
    std::array<uint8_t, 64> block = {};
    uint64_t total_len = 0;
    size_t block_len = 0;
};

// Canonical length-delimited digest serialization shared by every SHA-256 identity domain
// (little-endian fixed-width integers, u64-length-prefixed byte strings). Keeping one byte
// format here prevents silent drift between digest domains.
class llama_sha256_writer {
public:
    void bytes(const void * data, size_t size) {
        hash.update(data, size);
    }

    void u32(uint32_t value) {
        uint8_t data[4];
        llama_store_le_u32(data, value);
        bytes(data, sizeof(data));
    }

    void u64(uint64_t value) {
        uint8_t data[8];
        llama_store_le_u64(data, value);
        bytes(data, sizeof(data));
    }

    void string(const void * data, size_t size) {
        u64(size);
        if (size > 0) {
            bytes(data, size);
        }
    }

    std::array<uint8_t, 32> finish() {
        return hash.finish();
    }

private:
    llama_sha256 hash;
};
