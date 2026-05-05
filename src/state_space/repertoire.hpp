#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../core/ids.hpp"

namespace island_model {

class Repertoire {
public:
    static constexpr std::size_t word_bits_v = 64;

    Repertoire() = default;

    explicit Repertoire(std::size_t trait_count)
        : trait_count_(trait_count),
          words_(word_count_for(trait_count_), 0) {
        if (trait_count_ == 0) {
            throw std::invalid_argument("Repertoire: trait count must be > 0");
        }
    }

    [[nodiscard]] std::size_t trait_count() const noexcept {
        return trait_count_;
    }

    [[nodiscard]] bool contains(TraitId trait) const {
        check_trait(trait);
        const auto [word_index, bit_index] = locate(trait);
        return (words_[word_index] & bit_mask(bit_index)) != 0;
    }

    void add(TraitId trait) {
        check_trait(trait);
        const auto [word_index, bit_index] = locate(trait);
        words_[word_index] |= bit_mask(bit_index);
    }

    void remove(TraitId trait) {
        check_trait(trait);
        const auto [word_index, bit_index] = locate(trait);
        words_[word_index] &= ~bit_mask(bit_index);
    }

    [[nodiscard]] Repertoire with_added(TraitId trait) const {
        Repertoire copy = *this;
        copy.add(trait);
        return copy;
    }

    [[nodiscard]] bool empty() const noexcept {
        return std::ranges::all_of(words_, [](const std::uint64_t word) {
            return word == 0;
        });
    }

    [[nodiscard]] std::size_t size() const noexcept {
        std::size_t count = 0;
        for (const std::uint64_t word : words_) {
            count += static_cast<std::size_t>(std::popcount(word));
        }
        return count;
    }

    void clear() noexcept {
        std::fill(words_.begin(), words_.end(), 0);
    }

    [[nodiscard]] const std::vector<std::uint64_t>& words() const noexcept {
        return words_;
    }

    [[nodiscard]] bool operator==(const Repertoire& other) const noexcept {
        return trait_count_ == other.trait_count_ && words_ == other.words_;
    }

private:
    [[nodiscard]] static std::pair<std::size_t, std::size_t> locate(TraitId trait) noexcept {
        const auto t = static_cast<std::size_t>(trait);
        return {t / word_bits_v, t % word_bits_v};
    }

    [[nodiscard]] static std::uint64_t bit_mask(std::size_t bit_index) noexcept {
        return std::uint64_t{1} << bit_index;
    }

    [[nodiscard]] static std::size_t word_count_for(std::size_t trait_count) noexcept {
        return (trait_count + word_bits_v - 1) / word_bits_v;
    }

    void check_trait(TraitId trait) const {
        if (static_cast<std::size_t>(trait) >= trait_count_) {
            throw std::out_of_range("Repertoire: invalid trait id");
        }
    }

    std::size_t trait_count_{0};
    std::vector<std::uint64_t> words_{};
};

struct RepertoireHash {
    [[nodiscard]] std::size_t operator()(const Repertoire& r) const noexcept {
        const auto& w = r.words();

        std::size_t seed = r.trait_count();
        for (const std::uint64_t word : w) {
            hash_combine(seed, static_cast<std::size_t>(word));
        }
        return seed;
    }

private:
    static void hash_combine(std::size_t& seed, std::size_t value) noexcept {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }
};

} // namespace island_model
