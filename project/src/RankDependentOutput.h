#ifndef RANKDEPENDENTOUTPUT_H
#define RANKDEPENDENTOUTPUT_H

#include <tclap/CmdLineInterface.h>
#include <tclap/StdOutput.h>

class RankDependentOutput : public TCLAP::StdOutput {
private:
    int _rank;
    
public:
    RankDependentOutput(int rank) {
        _rank = rank;
    }
    
    virtual void usage(TCLAP::CmdLineInterface& c) {
        if (_rank == 0) TCLAP::StdOutput::usage(c);
    }
    
    virtual void version(TCLAP::CmdLineInterface& c) {
        if (_rank == 0) TCLAP::StdOutput::version(c);
    }
    
    virtual void failure(TCLAP::CmdLineInterface& c, TCLAP::ArgException& e) {
        if (_rank == 0) TCLAP::StdOutput::failure(c, e);
    }
};
#endif
