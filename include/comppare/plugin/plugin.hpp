/*

Copyright 2025 | Leong Fan FUNG | funglf | stanleyfunglf@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

*/
#pragma once
#include <concepts>
#include <utility>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>


namespace comppare::plugin
{
    template <class InTup, class OutTup>
    class Plugin
    {
    public:
        virtual ~Plugin() = default;
        virtual void initialize(int & /*argc*/, char ** /*argv*/) {}
        virtual void run() {}
    };

    template <template <class,class> class P, class InTup, class OutTup, class Func>
    concept ValidPlugin =
        requires { { P<InTup, OutTup>::instance() } -> std::same_as<std::shared_ptr<P<InTup, OutTup>>>; }
        &&
        requires(const std::string& name, Func&& user_fn, const InTup& inputs, OutTup& outputs)
        { std::declval<P<InTup, OutTup>&>().register_impl(name, user_fn, inputs, outputs); }
        &&
        std::derived_from<P<InTup, OutTup>, plugin::Plugin<InTup, OutTup>>;


    class PluginArgParser {
    public:
        explicit PluginArgParser(std::string header, bool strict_missing_value = false)
            : header_(std::move(header)), strict_(strict_missing_value) {}

        PluginArgParser(const PluginArgParser&) = delete;
        PluginArgParser& operator=(const PluginArgParser&) = delete;
        PluginArgParser(PluginArgParser&&) = default;
        PluginArgParser& operator=(PluginArgParser&&) = default;
        ~PluginArgParser() = default;

        [[nodiscard]] std::pair<int, char**> parse(int argc, char** argv) {
            args_.clear();
            cargv_.clear();

            if (argc <= 0 || argv == nullptr || argv[0] == nullptr) {
                args_.push_back(header_.empty() ? "program" : header_);
            } else {
                args_.emplace_back(argv[0]);
            }

            const std::string eq_prefix = header_ + "=";
            std::vector<std::string> tokens;

            for (int i = 1; i < argc; ++i) {
                if (!argv || argv[i] == nullptr) break;
                const std::string cur(argv[i]);

                if (starts_with(cur, eq_prefix)) {
                    const std::string value = cur.substr(eq_prefix.size());
                    append_tokens(tokens, value);
                } else if (cur == header_) {
                    if (i + 1 < argc && argv[i + 1] != nullptr) {
                        const std::string value(argv[++i]); // consume value
                        append_tokens(tokens, value);
                    } else if (strict_) {
                        throw std::invalid_argument(header_ + " requires a value");
                    }
                }
            }

            for (const auto& t : tokens) args_.push_back(t);

            cargv_.reserve(args_.size() + 1);
            for (const auto& s : args_) cargv_.push_back(const_cast<char*>(s.c_str()));
            cargv_.push_back(nullptr);

            return { static_cast<int>(args_.size()), cargv_.data() };
        }

        [[nodiscard]] int argc() const { return static_cast<int>(args_.size()); }
        [[nodiscard]] char** argv()    { return cargv_.data(); } // valid after parse()
        [[nodiscard]] const std::vector<std::string>& args() const { return args_; }

    private:
        std::string header_;
        bool strict_;
        std::vector<std::string> args_;
        std::vector<char*> cargv_;

        static bool starts_with(const std::string& s, const std::string& prefix) {
            return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
        }

        // shell-like split that respects quotes: R"(--x "a b" c)" -> {"--x","a b","c"}
        static std::vector<std::string> split_shell_like(const std::string& s) {
            std::vector<std::string> out;
            std::istringstream iss(s);
            std::string tok;
            while (iss >> std::quoted(tok)) out.push_back(tok);
            return out;
        }

        static void append_tokens(std::vector<std::string>& dst, const std::string& value) {
            auto toks = split_shell_like(value);
            dst.insert(dst.end(), toks.begin(), toks.end());
        }
    };

} // namespace comppare::plugin