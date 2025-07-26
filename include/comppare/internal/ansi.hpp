#pragma once
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <concepts>
#include <utility>

/*
ECMA-48 “Select Graphic Rendition” (SGR) sequences are defined here:
https://man7.org/linux/man-pages/man4/console_codes.4.html
*/
namespace comppare::internal::ansi
{
    template <typename T>
    concept Streamable =
        requires(std::ostream &os, T v) { { os << v } -> std::same_as<std::ostream&>; };

    template <Streamable T>
    class AnsiWrapper
    {
        const char *on_;  // "on" code
        const char *off_; // "off" code
        T val_;

    public:
        AnsiWrapper(const char *on, const char *off, T v)
            : on_(on), off_(off), val_(std::move(v)) {}

        friend std::ostream &operator<<(std::ostream &os, AnsiWrapper const &w)
        {
            std::ios saved(nullptr);
            saved.copyfmt(os);

            std::ostringstream tmp;
            tmp.copyfmt(os);
            tmp << w.val_;
            std::string body = std::move(tmp).str();

            os.width(0);
            os << w.on_ << body << w.off_;
            os.copyfmt(saved);
            return os;
        }
    };
    template <Streamable T>
    using AW = AnsiWrapper<std::decay_t<T>>;

// ── generator macro for each style ───────────────────
#define ANSI_DEFINE(NAME, ON, OFF)                                             \
    inline constexpr const char *NAME##_ON_CODE = "\033[" ON "m";              \
    inline constexpr const char *NAME##_OFF_CODE = "\033[" OFF "m";            \
    class NAME##_t                                                             \
    {                                                                          \
    public:                                                                    \
        friend std::ostream &operator<<(std::ostream &os, NAME##_t)            \
        {                                                                      \
            std::ios saved(nullptr);                                           \
            saved.copyfmt(os);                                                 \
            os.width(0);                                                       \
            os << NAME##_ON_CODE;                                              \
            os.copyfmt(saved);                                                 \
            return os;                                                         \
        }                                                                      \
        template <Streamable T>                                                \
        auto operator()(T &&v) const                                           \
        {                                                                      \
            return AW<T>(NAME##_ON_CODE, NAME##_OFF_CODE, std::forward<T>(v)); \
        }                                                                      \
    };                                                                         \
    inline constexpr NAME##_t NAME{};                                          \
    class NAME##_OFF_t                                                         \
    {                                                                          \
    public:                                                                    \
        friend std::ostream &operator<<(std::ostream &os, NAME##_OFF_t)        \
        {                                                                      \
            std::ios saved(nullptr);                                           \
            saved.copyfmt(os);                                                 \
            os.width(0);                                                       \
            os << NAME##_OFF_CODE;                                             \
            os.copyfmt(saved);                                                 \
            return os;                                                         \
        }                                                                      \
    };                                                                         \
    inline constexpr NAME##_OFF_t NAME##_OFF{};

    // ────────────────────────────────────────────────────────────────────────────
    //                          STYLE ATTRIBUTES
    // ────────────────────────────────────────────────────────────────────────────
    ANSI_DEFINE(RESET, "0", "0");
    ANSI_DEFINE(BOLD, "1", "22");
    ANSI_DEFINE(DIM, "2", "22");
    ANSI_DEFINE(ITALIC, "3", "23");
    ANSI_DEFINE(UNDERLINE, "4", "24");
    ANSI_DEFINE(BLINK, "5", "25");
    ANSI_DEFINE(REVERSE, "7", "27");
    ANSI_DEFINE(HIDDEN, "8", "28");
    ANSI_DEFINE(STRIKE, "9", "29");

    // ────────────────────────────────────────────────────────────────────────────
    //                        FOREGROUND  (30-37 / 90-97)
    // ────────────────────────────────────────────────────────────────────────────
    ANSI_DEFINE(BLACK, "30", "39");
    ANSI_DEFINE(RED, "31", "39");
    ANSI_DEFINE(GREEN, "32", "39");
    ANSI_DEFINE(YELLOW, "33", "39");
    ANSI_DEFINE(BLUE, "34", "39");
    ANSI_DEFINE(MAGENTA, "35", "39");
    ANSI_DEFINE(CYAN, "36", "39");
    ANSI_DEFINE(WHITE, "37", "39");

    ANSI_DEFINE(BRIGHT_BLACK, "90", "39");
    ANSI_DEFINE(BRIGHT_RED, "91", "39");
    ANSI_DEFINE(BRIGHT_GREEN, "92", "39");
    ANSI_DEFINE(BRIGHT_YELLOW, "93", "39");
    ANSI_DEFINE(BRIGHT_BLUE, "94", "39");
    ANSI_DEFINE(BRIGHT_MAGENTA, "95", "39");
    ANSI_DEFINE(BRIGHT_CYAN, "96", "39");
    ANSI_DEFINE(BRIGHT_WHITE, "97", "39");

    // ────────────────────────────────────────────────────────────────────────────
    //                        BACKGROUND (40-47 / 100-107)
    // ────────────────────────────────────────────────────────────────────────────
    ANSI_DEFINE(BG_BLACK, "40", "49");
    ANSI_DEFINE(BG_RED, "41", "49");
    ANSI_DEFINE(BG_GREEN, "42", "49");
    ANSI_DEFINE(BG_YELLOW, "43", "49");
    ANSI_DEFINE(BG_BLUE, "44", "49");
    ANSI_DEFINE(BG_MAGENTA, "45", "49");
    ANSI_DEFINE(BG_CYAN, "46", "49");
    ANSI_DEFINE(BG_WHITE, "47", "49");

    ANSI_DEFINE(BG_BRIGHT_BLACK, "100", "49");
    ANSI_DEFINE(BG_BRIGHT_RED, "101", "49");
    ANSI_DEFINE(BG_BRIGHT_GREEN, "102", "49");
    ANSI_DEFINE(BG_BRIGHT_YELLOW, "103", "49");
    ANSI_DEFINE(BG_BRIGHT_BLUE, "104", "49");
    ANSI_DEFINE(BG_BRIGHT_MAGENTA, "105", "49");
    ANSI_DEFINE(BG_BRIGHT_CYAN, "106", "49");
    ANSI_DEFINE(BG_BRIGHT_WHITE, "107", "49");

#undef ANSI_DEFINE
}