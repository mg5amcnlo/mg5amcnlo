//==========================================================================
// Copyright (C) 2023-2026 CERN
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Written originally by: Z. Wettersten (Jan 2024) for the MG5aMC CUDACPP plugin.
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
//==========================================================================
// This file has been automatically generated for C++ Standalone by
%(info_lines)s
//==========================================================================
//==========================================================================
// Driver for reweighting events for processes
%(multiprocess_lines)s
//--------------------------------------------------------------------------

#include "rwgt_instance.h"
#include <cstdlib>
#include <typeinfo>
#include <memory>
%(include_lines)s

    int usage(char *argv0, int ret = 1)
{
    std::cout << "Usage: " << argv0
              << " [--lhefile=\"/YOUR/PATH/HERE\"|-lhe=\"/YOUR/PATH/HERE\"] [--rwgtcard=/YOUR/PATH/HERE|-rwgt=\"/YOUR/PATH/HERE\"]\n"
              << "[--output=/YOUR/PATH/HERE\"|-out=\"/YOUR/PATH/HERE\"]\n"
              << "[--param_card=/YOUR/PATH/HERE\"|-slha=\"/YOUR/PATH/HERE\"]\n";
    std::cout << "\n";
    std::cout << "The LHE file path should be with respect to the directory you are running\n";
    std::cout << "this program from, and similarly the rwgt_card should be as well.\n";
    return ret;
}

void writeRwgtCsv(std::string path, std::shared_ptr<std::vector<std::string>> names, std::shared_ptr<std::vector<double>> xSecs, std::shared_ptr<std::vector<double>> errXSecs)
{
    std::ofstream outFile;
    outFile.open(path);
    if (!outFile.is_open())
        throw std::runtime_error("Failed to open output file for writing.");
    if (names->size() != xSecs->size() || names->size() != errXSecs->size())
        throw std::runtime_error("Mismatch in number of processes, cross-sections, and errors when logging results.");
    for (size_t k = 0; k < names->size(); ++k)
    {
        outFile << names->at(k) << ", " << xSecs->at(k) << ", " << errXSecs->at(k) << "\n";
    }
    outFile.close();
    return;
}

void writeRwgtCsv(std::string path, std::vector<std::string> names, std::vector<double> xSecs, std::vector<double> errXSecs)
{
    std::ofstream outFile;
    outFile.open(path);
    if (!outFile.is_open())
        throw std::runtime_error("Failed to open output file for writing.");
    if (names.size() != xSecs.size() || names.size() != errXSecs.size())
        throw std::runtime_error("Mismatch in number of processes, cross-sections, and errors when logging results.");
    for (size_t k = 0; k < names.size(); ++k)
    {
        outFile << names.at(k) << ", " << xSecs.at(k) << ", " << errXSecs.at(k) << "\n";
    }
    outFile.close();
    return;
}

int main(int argc, char **argv)
{

    std::string banner =
        "#################################################\n"
        "#    ___  ___          _ _  ______              #\n"
        "#    |  \\/  |         | | | | ___ \\             #\n"
        "#    | .  . | __ _  __| | |_| |_/ /_____  __    #\n"
        "#    | |\\/| |/ _` |/ _` | __|    // _ \\ \\/ /    #\n"
        "#    | |  | | (_| | (_| | |_| |\\ \\  __/>  <     #\n"
        "#    \\_|  |_/\\__,_|\\__,_|\\__\\_| \\_\\___/_/\\_\\    #\n"
        "#                                               #\n"
        "#        Data-parallel event reweighting        #\n"
        "#             in MadGraph5_aMC@NLO              #\n"
        "#                                               #\n"
        "#################################################\n";

    std::cout << banner;
    std::cout << "Starting MadtRex driver...\n";
    std::string lheFilePath;
    std::string rwgtCardPath;
    std::string outputPath;
    std::string slhaPath;
    size_t nb_threads = 1;
    size_t batch_size = 32;

    if (argc < 2)
    {
        return usage(argv[0]);
    }

    for (int i = 1; i < argc; i++)
    {
        auto currArg = std::string(argv[i]);
        if (currArg.substr(0, 9) == "--lhefile" || currArg.substr(0, 4) == "-lhe")
        {
            lheFilePath = currArg.substr(currArg.find("=") + 1);
        }
        else if (currArg.substr(0, 10) == "--rwgtcard" || currArg.substr(0, 5) == "-rwgt")
        {
            rwgtCardPath = currArg.substr(currArg.find("=") + 1);
        }
        else if (currArg.substr(0, 8) == "--output" || currArg.substr(0, 4) == "-out")
        {
            outputPath = currArg.substr(currArg.find("=") + 1);
        }
        else if (currArg.substr(0, 12) == "--param_card" || currArg.substr(0, 5) == "-slha")
        {
            slhaPath = currArg.substr(currArg.find("=") + 1);
        }
        else if (currArg.substr(0, 10) == "--nb_threads" || currArg.substr(0, 3) == "-nt")
        {
            nb_threads = std::stoi(currArg.substr(currArg.find("=") + 1));
        }
        else if (currArg.substr(0, 12) == "--batch_size" || currArg.substr(0, 5) == "-warp")
        {
            batch_size = std::stoi(currArg.substr(currArg.find("=") + 1));
        }
        else
        {
            return usage(argv[0]);
        }
    }

    if (lheFilePath.empty() || rwgtCardPath.empty())
    {
        return usage(argv[0]);
    }

    std::string currPath = argv[0];

    size_t slashPos = currPath.find_last_of("/");
    bool onWindows = false;
    if (slashPos == std::string::npos)
    {
        slashPos = currPath.find_last_of("\\");
        onWindows = true;
    }
    if (slashPos == std::string::npos)
        throw std::runtime_error("Failed to determine current working directory -- need to know where program is run from to identify where to pull and push param_card.dat.");

    if (slhaPath.empty())
    {
        if (onWindows)
        {
            if (currPath.substr(currPath.find_last_of("\\", slashPos - 1) + 1, 2) == "P1")
            {
                slhaPath = "..\\..\\Cards\\param_card.dat";
            }
            else if (currPath.substr(currPath.find_last_of("\\", slashPos - 1) + 1, 3) == "Sub")
            {
                slhaPath = "..\\Cards\\param_card.dat";
            }
            else
            {
                slhaPath = "\\Cards\\param_card.dat";
            }
        }
        else
        {
            if (currPath.substr(currPath.find_last_of("/", slashPos - 1) + 1, 2) == "P1")
            {
                slhaPath = "../../Cards/param_card.dat";
            }
            else if (currPath.substr(currPath.find_last_of("/", slashPos - 1) + 1, 3) == "Sub")
            {
                slhaPath = "../Cards/param_card.dat";
            }
            else
            {
                slhaPath = "/Cards/param_card.dat";
            }
        }
    }

    static std::vector<std::shared_ptr<REX::tea::procReweightor>> rwgtRun = {%(make_rwgt)s};

    auto rwgt_runner = REX::tea::param_rwgt(REX::load_lhef(lheFilePath), rwgtRun);
    rwgt_runner.read_slha_rwgt(slhaPath, rwgtCardPath);
    rwgt_runner.pool_threads = nb_threads;

    rwgt_runner.run();

    std::cout << "\nReweighting procedure finished.\n";

    std::ofstream lhe_out(outputPath);
    if (!lhe_out)
        throw std::runtime_error("Failed to open output LHE file for writing.");
    rwgt_runner.print(lhe_out, true);

    std::cout << "Reweighted LHE file written to " << outputPath << ".\n";

    writeRwgtCsv("rwgt_results.csv", *rwgt_runner.weight_ids, rwgt_runner.rwgt_xSec, rwgt_runner.rwgt_xErr);

    return 0;
}
