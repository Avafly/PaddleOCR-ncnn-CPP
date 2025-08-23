#ifndef LOGGING_H_
#define LOGGING_H_

#include "plog/Log.h"
#include "plog/Initializers/ConsoleInitializer.h"

namespace plog
{
    class MyFormatter
    {
    public:
        static util::nstring header()
        {
            return util::nstring();
        }

        static util::nstring format(const Record &record)
        {
            util::nostringstream ss;
            util::nstring severity_str = severityToString(record.getSeverity());
            if (!severity_str.empty())
                ss << PLOG_NSTR("[") << severity_str[0] << PLOG_NSTR("] ");
            ss << PLOG_NSTR("[") << record.getFunc() << PLOG_NSTR("@") << record.getLine() << PLOG_NSTR("] ");
            ss << record.getMessage() << PLOG_NSTR("\n");

            return ss.str();
        }
    };
}

#endif  // LOGGING_H_