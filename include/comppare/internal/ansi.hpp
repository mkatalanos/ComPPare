#pragma once
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <concepts>
#include <utility>

#include <comppare/internal/concepts.hpp>

/*
ECMA-48 “Select Graphic Rendition” (SGR) sequences are defined here:
https://man7.org/linux/man-pages/man4/console_codes.4.html
*/
namespace comppare::internal::ansi
{
    /*
    Wrapper Class to apply ANSI styles to any streamable type. [template <Streamable T>]
    It wraps the value and applies the "ON" and "OFF" ANSI codes around it
    */
    template <comppare::internal::concepts::Streamable T>
    class AnsiWrapper
    {
        const char *on_;  // "on" code
        const char *off_; // "off" code
        T val_;           // the value to be wrapped

    public:
        AnsiWrapper(const char *on, const char *off, T v)
            : on_(on), off_(off), val_(std::move(v)) {}

        /*
        Overloaded operator<< to stream the value with ANSI codes.
        It takes in the output stream state and applies it to the value.
        */
        friend std::ostream &operator<<(std::ostream &os, AnsiWrapper const &w)
        {
            // Save the current state of the stream
            std::ios saved(nullptr);
            saved.copyfmt(os);

            // convert the value into a string
            std::ostringstream tmp;
            tmp.copyfmt(os);
            tmp << w.val_;
            std::string body = std::move(tmp).str();

            // apply the ANSI codes
            os.width(0);
            os << w.on_ << body << w.off_;

            // Restore the original state of the stream -- formatting, precision, etc.
            os.copyfmt(saved);
            return os;
        }
    };
    template <comppare::internal::concepts::Streamable T>
    using AW = AnsiWrapper<std::decay_t<T>>;

/*
Generator Macro for each ANSI style and color with the AnsiWrapper class.
*/
#define ANSI_DEFINE(NAME, ON, OFF)                                                                                                  \
    /* Actual ANSI codes for the style/color (eg: BLACK_ON_CODE = "\033[30m") */                                                    \
    inline constexpr const char *NAME##_ON_CODE = "\033[" ON "m";                                                                   \
    inline constexpr const char *NAME##_OFF_CODE = "\033[" OFF "m";                                                                 \
                                                                                                                                    \
    /* Wrapper class for the different usage patterns */                                                                            \
    /* https://learn.microsoft.com/en-us/cpp/standard-library/overloading-the-output-operator-for-your-own-classes?view=msvc-170 */ \
    class NAME##ON_t                                                                                                                \
    {                                                                                                                               \
    public:                                                                                                                         \
        /* Usage: std::cout << comppare::internal::ansi::BOLD << "Hello world"; */                                                  \
        friend std::ostream &operator<<(std::ostream &os, NAME##ON_t)                                                               \
        {                                                                                                                           \
            /* Save the current state of the stream and apply the ANSI code */                                                      \
            std::ios saved(nullptr);                                                                                                \
            saved.copyfmt(os);                                                                                                      \
            os.width(0);                                                                                                            \
            os << NAME##_ON_CODE;                                                                                                   \
            os.copyfmt(saved);                                                                                                      \
            return os;                                                                                                              \
        }                                                                                                                           \
                                                                                                                                    \
        /* Usage: std::cout << comppare::internal::ansi::BOLD("Hello world") */                                                     \
        template <comppare::internal::concepts::Streamable T>                                                                                                     \
        auto operator()(T &&v) const                                                                                                \
        {                                                                                                                           \
            return AW<T>(NAME##_ON_CODE, NAME##_OFF_CODE, std::forward<T>(v));                                                      \
        }                                                                                                                           \
    };                                                                                                                              \
    /* Instance of the ON class */                                                                                                  \
    inline constexpr NAME##ON_t NAME{};                                                                                             \
    class NAME##_OFF_t                                                                                                              \
    {                                                                                                                               \
    public:                                                                                                                         \
        /* Usage: std::cout << ... << comppare::internal::ansi::BOLD_OFF; */                                                        \
        friend std::ostream &operator<<(std::ostream &os, NAME##_OFF_t)                                                             \
        {                                                                                                                           \
            /* Save the current state of the stream and apply the ANSI code */                                                      \
            std::ios saved(nullptr);                                                                                                \
            saved.copyfmt(os);                                                                                                      \
            os.width(0);                                                                                                            \
            os << NAME##_OFF_CODE;                                                                                                  \
            os.copyfmt(saved);                                                                                                      \
            return os;                                                                                                              \
        }                                                                                                                           \
    };                                                                                                                              \
    /* Instance of the OFF class */                                                                                                 \
    inline constexpr NAME##_OFF_t NAME##_OFF{};

    /*
    Below are the ANSI escape codes for various styles and colors.
    Each style has a corresponding ON and OFF code.
    */

    /* STYLES 0-9 */
    ANSI_DEFINE(RESET,      "0", "0");
    ANSI_DEFINE(BOLD,       "1", "22");
    ANSI_DEFINE(DIM,        "2", "22");
    ANSI_DEFINE(ITALIC,     "3", "23");
    ANSI_DEFINE(UNDERLINE,  "4", "24");
    ANSI_DEFINE(BLINK,      "5", "25");
    ANSI_DEFINE(REVERSE,    "7", "27");
    ANSI_DEFINE(HIDDEN,     "8", "28");
    ANSI_DEFINE(STRIKE,     "9", "29");

    /* FOREGROUND (TEXT) COLOURS 30-37/90-97 */
    ANSI_DEFINE(BLACK,      "30", "39");
    ANSI_DEFINE(RED,        "31", "39");
    ANSI_DEFINE(GREEN,      "32", "39");
    ANSI_DEFINE(YELLOW,     "33", "39");
    ANSI_DEFINE(BLUE,       "34", "39");
    ANSI_DEFINE(MAGENTA,    "35", "39");
    ANSI_DEFINE(CYAN,       "36", "39");
    ANSI_DEFINE(WHITE,      "37", "39");

    ANSI_DEFINE(BRIGHT_BLACK,   "90", "39");
    ANSI_DEFINE(BRIGHT_RED,     "91", "39");
    ANSI_DEFINE(BRIGHT_GREEN,   "92", "39");
    ANSI_DEFINE(BRIGHT_YELLOW,  "93", "39");
    ANSI_DEFINE(BRIGHT_BLUE,    "94", "39");
    ANSI_DEFINE(BRIGHT_MAGENTA, "95", "39");
    ANSI_DEFINE(BRIGHT_CYAN,    "96", "39");
    ANSI_DEFINE(BRIGHT_WHITE,   "97", "39");

    /* BACKGROUND COLOURS (40-47 / 100-107) */
    ANSI_DEFINE(BG_BLACK,   "40", "49");
    ANSI_DEFINE(BG_RED,     "41", "49");
    ANSI_DEFINE(BG_GREEN,   "42", "49");
    ANSI_DEFINE(BG_YELLOW,  "43", "49");
    ANSI_DEFINE(BG_BLUE,    "44", "49");
    ANSI_DEFINE(BG_MAGENTA, "45", "49");
    ANSI_DEFINE(BG_CYAN,    "46", "49");
    ANSI_DEFINE(BG_WHITE,   "47", "49");

    ANSI_DEFINE(BG_BRIGHT_BLACK,    "100", "49");
    ANSI_DEFINE(BG_BRIGHT_RED,      "101", "49");
    ANSI_DEFINE(BG_BRIGHT_GREEN,    "102", "49");
    ANSI_DEFINE(BG_BRIGHT_YELLOW,   "103", "49");
    ANSI_DEFINE(BG_BRIGHT_BLUE,     "104", "49");
    ANSI_DEFINE(BG_BRIGHT_MAGENTA,  "105", "49");
    ANSI_DEFINE(BG_BRIGHT_CYAN,     "106", "49");
    ANSI_DEFINE(BG_BRIGHT_WHITE,    "107", "49");

#undef ANSI_DEFINE
}