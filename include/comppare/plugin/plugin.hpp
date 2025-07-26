#pragma once

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
} // namespace comppare::plugin