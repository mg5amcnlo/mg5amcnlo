#include <Pythia8/Basics.h>
#include <Pythia8/BeamParticle.h>
#include <Pythia8/Event.h>
#include <Pythia8/FragmentationFlavZpT.h>
#include <Pythia8/HadronWidths.h>
#include <Pythia8/Info.h>
#include <Pythia8/LHEF3.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/PartonDistributions.h>
#include <Pythia8/PartonSystems.h>
#include <Pythia8/ResonanceWidths.h>
#include <Pythia8/Settings.h>
#include <Pythia8/SigmaTotal.h>
#include <Pythia8/StandardModel.h>
#include <Pythia8/SusyCouplings.h>
#include <Pythia8/SusyLesHouches.h>
#include <Pythia8/Weights.h>
#include <complex>
#include <functional>
#include <istream>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <sstream> // __str__
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <Pythia8/UserHooks.h>
#include <Pythia8/HIUserHooks.h>
#include <Pythia8/HeavyIons.h>
#include <Pythia8/BeamShape.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_Pythia8_Info(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::Info file:Pythia8/Info.h line:42
		pybind11::class_<Pythia8::Info, std::shared_ptr<Pythia8::Info>> cl(M("Pythia8"), "Info", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::Info(); } ) );
		cl.def( pybind11::init<bool>(), pybind11::arg("") );

		cl.def( pybind11::init( [](Pythia8::Info const &o){ return new Pythia8::Info(o); } ) );
		cl.def_readwrite("userHooksPtr", &Pythia8::Info::userHooksPtr);
		cl.def_readwrite("LHEFversionSave", &Pythia8::Info::LHEFversionSave);
		cl.def_readwrite("sigmaLHEFSave", &Pythia8::Info::sigmaLHEFSave);
		cl.def_readwrite("hasOwnEventAttributes", &Pythia8::Info::hasOwnEventAttributes);
		cl.def_readwrite("weights_detailed_vector", &Pythia8::Info::weights_detailed_vector);
		cl.def_readwrite("eventWeightLHEF", &Pythia8::Info::eventWeightLHEF);
		cl.def_readwrite("idASave", &Pythia8::Info::idASave);
		cl.def_readwrite("idBSave", &Pythia8::Info::idBSave);
		cl.def_readwrite("pzASave", &Pythia8::Info::pzASave);
		cl.def_readwrite("eASave", &Pythia8::Info::eASave);
		cl.def_readwrite("mASave", &Pythia8::Info::mASave);
		cl.def_readwrite("pzBSave", &Pythia8::Info::pzBSave);
		cl.def_readwrite("eBSave", &Pythia8::Info::eBSave);
		cl.def_readwrite("mBSave", &Pythia8::Info::mBSave);
		cl.def_readwrite("eCMSave", &Pythia8::Info::eCMSave);
		cl.def_readwrite("sSave", &Pythia8::Info::sSave);
		cl.def_readwrite("lowPTmin", &Pythia8::Info::lowPTmin);
		cl.def_readwrite("nTry", &Pythia8::Info::nTry);
		cl.def_readwrite("nSel", &Pythia8::Info::nSel);
		cl.def_readwrite("nAcc", &Pythia8::Info::nAcc);
		cl.def_readwrite("sigGen", &Pythia8::Info::sigGen);
		cl.def_readwrite("sigErr", &Pythia8::Info::sigErr);
		cl.def_readwrite("wtAccSum", &Pythia8::Info::wtAccSum);
		cl.def_readwrite("procNameM", &Pythia8::Info::procNameM);
		cl.def_readwrite("nTryM", &Pythia8::Info::nTryM);
		cl.def_readwrite("nSelM", &Pythia8::Info::nSelM);
		cl.def_readwrite("nAccM", &Pythia8::Info::nAccM);
		cl.def_readwrite("sigGenM", &Pythia8::Info::sigGenM);
		cl.def_readwrite("sigErrM", &Pythia8::Info::sigErrM);
		cl.def_readwrite("lhaStrategySave", &Pythia8::Info::lhaStrategySave);
		cl.def_readwrite("a0MPISave", &Pythia8::Info::a0MPISave);
		cl.def_readwrite("isRes", &Pythia8::Info::isRes);
		cl.def_readwrite("isDiffA", &Pythia8::Info::isDiffA);
		cl.def_readwrite("isDiffB", &Pythia8::Info::isDiffB);
		cl.def_readwrite("isDiffC", &Pythia8::Info::isDiffC);
		cl.def_readwrite("isND", &Pythia8::Info::isND);
		cl.def_readwrite("isLH", &Pythia8::Info::isLH);
		cl.def_readwrite("bIsSet", &Pythia8::Info::bIsSet);
		cl.def_readwrite("evolIsSet", &Pythia8::Info::evolIsSet);
		cl.def_readwrite("atEOF", &Pythia8::Info::atEOF);
		cl.def_readwrite("isVal1", &Pythia8::Info::isVal1);
		cl.def_readwrite("isVal2", &Pythia8::Info::isVal2);
		cl.def_readwrite("hasHistorySave", &Pythia8::Info::hasHistorySave);
		cl.def_readwrite("abortPartonLevel", &Pythia8::Info::abortPartonLevel);
		cl.def_readwrite("isHardDiffA", &Pythia8::Info::isHardDiffA);
		cl.def_readwrite("isHardDiffB", &Pythia8::Info::isHardDiffB);
		cl.def_readwrite("hasUnresBeams", &Pythia8::Info::hasUnresBeams);
		cl.def_readwrite("hasPomPsys", &Pythia8::Info::hasPomPsys);
		cl.def_readwrite("codeSave", &Pythia8::Info::codeSave);
		cl.def_readwrite("nFinalSave", &Pythia8::Info::nFinalSave);
		cl.def_readwrite("nTotal", &Pythia8::Info::nTotal);
		cl.def_readwrite("nMPISave", &Pythia8::Info::nMPISave);
		cl.def_readwrite("nISRSave", &Pythia8::Info::nISRSave);
		cl.def_readwrite("nFSRinProcSave", &Pythia8::Info::nFSRinProcSave);
		cl.def_readwrite("nFSRinResSave", &Pythia8::Info::nFSRinResSave);
		cl.def_readwrite("bMPISave", &Pythia8::Info::bMPISave);
		cl.def_readwrite("enhanceMPISave", &Pythia8::Info::enhanceMPISave);
		cl.def_readwrite("enhanceMPIavgSave", &Pythia8::Info::enhanceMPIavgSave);
		cl.def_readwrite("bMPIoldSave", &Pythia8::Info::bMPIoldSave);
		cl.def_readwrite("enhanceMPIoldSave", &Pythia8::Info::enhanceMPIoldSave);
		cl.def_readwrite("enhanceMPIoldavgSave", &Pythia8::Info::enhanceMPIoldavgSave);
		cl.def_readwrite("pTmaxMPISave", &Pythia8::Info::pTmaxMPISave);
		cl.def_readwrite("pTmaxISRSave", &Pythia8::Info::pTmaxISRSave);
		cl.def_readwrite("pTmaxFSRSave", &Pythia8::Info::pTmaxFSRSave);
		cl.def_readwrite("pTnowSave", &Pythia8::Info::pTnowSave);
		cl.def_readwrite("zNowISRSave", &Pythia8::Info::zNowISRSave);
		cl.def_readwrite("pT2NowISRSave", &Pythia8::Info::pT2NowISRSave);
		cl.def_readwrite("xPomA", &Pythia8::Info::xPomA);
		cl.def_readwrite("xPomB", &Pythia8::Info::xPomB);
		cl.def_readwrite("tPomA", &Pythia8::Info::tPomA);
		cl.def_readwrite("tPomB", &Pythia8::Info::tPomB);
		cl.def_readwrite("nameSave", &Pythia8::Info::nameSave);
		cl.def_readwrite("codeMPISave", &Pythia8::Info::codeMPISave);
		cl.def_readwrite("iAMPISave", &Pythia8::Info::iAMPISave);
		cl.def_readwrite("iBMPISave", &Pythia8::Info::iBMPISave);
		cl.def_readwrite("pTMPISave", &Pythia8::Info::pTMPISave);
		cl.def_readwrite("eMPISave", &Pythia8::Info::eMPISave);
		cl.def_readwrite("isVMDstateAEvent", &Pythia8::Info::isVMDstateAEvent);
		cl.def_readwrite("isVMDstateBEvent", &Pythia8::Info::isVMDstateBEvent);
		cl.def_readwrite("gammaModeEvent", &Pythia8::Info::gammaModeEvent);
		cl.def_readwrite("idVMDASave", &Pythia8::Info::idVMDASave);
		cl.def_readwrite("idVMDBSave", &Pythia8::Info::idVMDBSave);
		cl.def_readwrite("x1GammaSave", &Pythia8::Info::x1GammaSave);
		cl.def_readwrite("x2GammaSave", &Pythia8::Info::x2GammaSave);
		cl.def_readwrite("Q2Gamma1Save", &Pythia8::Info::Q2Gamma1Save);
		cl.def_readwrite("Q2Gamma2Save", &Pythia8::Info::Q2Gamma2Save);
		cl.def_readwrite("eCMsubSave", &Pythia8::Info::eCMsubSave);
		cl.def_readwrite("thetaLepton1", &Pythia8::Info::thetaLepton1);
		cl.def_readwrite("thetaLepton2", &Pythia8::Info::thetaLepton2);
		cl.def_readwrite("sHatNewSave", &Pythia8::Info::sHatNewSave);
		cl.def_readwrite("mVMDASave", &Pythia8::Info::mVMDASave);
		cl.def_readwrite("mVMDBSave", &Pythia8::Info::mVMDBSave);
		cl.def_readwrite("scaleVMDASave", &Pythia8::Info::scaleVMDASave);
		cl.def_readwrite("scaleVMDBSave", &Pythia8::Info::scaleVMDBSave);
		cl.def_readwrite("messages", &Pythia8::Info::messages);
		cl.def_readwrite("headers", &Pythia8::Info::headers);
		cl.def_readwrite("headerBlock", &Pythia8::Info::headerBlock);
		cl.def_readwrite("eventComments", &Pythia8::Info::eventComments);
		cl.def_readwrite("plugins", &Pythia8::Info::plugins);
		cl.def_readwrite("weakModes", &Pythia8::Info::weakModes);
		cl.def_readwrite("weak2to2lines", &Pythia8::Info::weak2to2lines);
		cl.def_readwrite("weakMomenta", &Pythia8::Info::weakMomenta);
		cl.def_readwrite("weakDipoles", &Pythia8::Info::weakDipoles);
		cl.def("assign", (class Pythia8::Info & (Pythia8::Info::*)(const class Pythia8::Info &)) &Pythia8::Info::operator=, "C++: Pythia8::Info::operator=(const class Pythia8::Info &) --> class Pythia8::Info &", pybind11::return_value_policy::reference, pybind11::arg(""));
		cl.def("list", (void (Pythia8::Info::*)() const) &Pythia8::Info::list, "C++: Pythia8::Info::list() const --> void");
		cl.def("idA", (int (Pythia8::Info::*)() const) &Pythia8::Info::idA, "C++: Pythia8::Info::idA() const --> int");
		cl.def("idB", (int (Pythia8::Info::*)() const) &Pythia8::Info::idB, "C++: Pythia8::Info::idB() const --> int");
		cl.def("pzA", (double (Pythia8::Info::*)() const) &Pythia8::Info::pzA, "C++: Pythia8::Info::pzA() const --> double");
		cl.def("pzB", (double (Pythia8::Info::*)() const) &Pythia8::Info::pzB, "C++: Pythia8::Info::pzB() const --> double");
		cl.def("eA", (double (Pythia8::Info::*)() const) &Pythia8::Info::eA, "C++: Pythia8::Info::eA() const --> double");
		cl.def("eB", (double (Pythia8::Info::*)() const) &Pythia8::Info::eB, "C++: Pythia8::Info::eB() const --> double");
		cl.def("mA", (double (Pythia8::Info::*)() const) &Pythia8::Info::mA, "C++: Pythia8::Info::mA() const --> double");
		cl.def("mB", (double (Pythia8::Info::*)() const) &Pythia8::Info::mB, "C++: Pythia8::Info::mB() const --> double");
		cl.def("eCM", (double (Pythia8::Info::*)() const) &Pythia8::Info::eCM, "C++: Pythia8::Info::eCM() const --> double");
		cl.def("s", (double (Pythia8::Info::*)() const) &Pythia8::Info::s, "C++: Pythia8::Info::s() const --> double");
		cl.def("tooLowPTmin", (bool (Pythia8::Info::*)() const) &Pythia8::Info::tooLowPTmin, "C++: Pythia8::Info::tooLowPTmin() const --> bool");
		cl.def("name", (std::string (Pythia8::Info::*)() const) &Pythia8::Info::name, "C++: Pythia8::Info::name() const --> std::string");
		cl.def("code", (int (Pythia8::Info::*)() const) &Pythia8::Info::code, "C++: Pythia8::Info::code() const --> int");
		cl.def("nFinal", (int (Pythia8::Info::*)() const) &Pythia8::Info::nFinal, "C++: Pythia8::Info::nFinal() const --> int");
		cl.def("isResolved", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isResolved, "C++: Pythia8::Info::isResolved() const --> bool");
		cl.def("isDiffractiveA", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isDiffractiveA, "C++: Pythia8::Info::isDiffractiveA() const --> bool");
		cl.def("isDiffractiveB", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isDiffractiveB, "C++: Pythia8::Info::isDiffractiveB() const --> bool");
		cl.def("isDiffractiveC", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isDiffractiveC, "C++: Pythia8::Info::isDiffractiveC() const --> bool");
		cl.def("isNonDiffractive", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isNonDiffractive, "C++: Pythia8::Info::isNonDiffractive() const --> bool");
		cl.def("isElastic", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isElastic, "C++: Pythia8::Info::isElastic() const --> bool");
		cl.def("isMinBias", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isMinBias, "C++: Pythia8::Info::isMinBias() const --> bool");
		cl.def("isLHA", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isLHA, "C++: Pythia8::Info::isLHA() const --> bool");
		cl.def("atEndOfFile", (bool (Pythia8::Info::*)() const) &Pythia8::Info::atEndOfFile, "C++: Pythia8::Info::atEndOfFile() const --> bool");
		cl.def("hasSub", [](Pythia8::Info const &o) -> bool { return o.hasSub(); }, "");
		cl.def("hasSub", (bool (Pythia8::Info::*)(int) const) &Pythia8::Info::hasSub, "C++: Pythia8::Info::hasSub(int) const --> bool", pybind11::arg("i"));
		cl.def("nameSub", [](Pythia8::Info const &o) -> std::string { return o.nameSub(); }, "");
		cl.def("nameSub", (std::string (Pythia8::Info::*)(int) const) &Pythia8::Info::nameSub, "C++: Pythia8::Info::nameSub(int) const --> std::string", pybind11::arg("i"));
		cl.def("codeSub", [](Pythia8::Info const &o) -> int { return o.codeSub(); }, "");
		cl.def("codeSub", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::codeSub, "C++: Pythia8::Info::codeSub(int) const --> int", pybind11::arg("i"));
		cl.def("nFinalSub", [](Pythia8::Info const &o) -> int { return o.nFinalSub(); }, "");
		cl.def("nFinalSub", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::nFinalSub, "C++: Pythia8::Info::nFinalSub(int) const --> int", pybind11::arg("i"));
		cl.def("id1", [](Pythia8::Info const &o) -> int { return o.id1(); }, "");
		cl.def("id1", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::id1, "C++: Pythia8::Info::id1(int) const --> int", pybind11::arg("i"));
		cl.def("id2", [](Pythia8::Info const &o) -> int { return o.id2(); }, "");
		cl.def("id2", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::id2, "C++: Pythia8::Info::id2(int) const --> int", pybind11::arg("i"));
		cl.def("x1", [](Pythia8::Info const &o) -> double { return o.x1(); }, "");
		cl.def("x1", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::x1, "C++: Pythia8::Info::x1(int) const --> double", pybind11::arg("i"));
		cl.def("x2", [](Pythia8::Info const &o) -> double { return o.x2(); }, "");
		cl.def("x2", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::x2, "C++: Pythia8::Info::x2(int) const --> double", pybind11::arg("i"));
		cl.def("y", [](Pythia8::Info const &o) -> double { return o.y(); }, "");
		cl.def("y", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::y, "C++: Pythia8::Info::y(int) const --> double", pybind11::arg("i"));
		cl.def("tau", [](Pythia8::Info const &o) -> double { return o.tau(); }, "");
		cl.def("tau", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::tau, "C++: Pythia8::Info::tau(int) const --> double", pybind11::arg("i"));
		cl.def("id1pdf", [](Pythia8::Info const &o) -> int { return o.id1pdf(); }, "");
		cl.def("id1pdf", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::id1pdf, "C++: Pythia8::Info::id1pdf(int) const --> int", pybind11::arg("i"));
		cl.def("id2pdf", [](Pythia8::Info const &o) -> int { return o.id2pdf(); }, "");
		cl.def("id2pdf", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::id2pdf, "C++: Pythia8::Info::id2pdf(int) const --> int", pybind11::arg("i"));
		cl.def("x1pdf", [](Pythia8::Info const &o) -> double { return o.x1pdf(); }, "");
		cl.def("x1pdf", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::x1pdf, "C++: Pythia8::Info::x1pdf(int) const --> double", pybind11::arg("i"));
		cl.def("x2pdf", [](Pythia8::Info const &o) -> double { return o.x2pdf(); }, "");
		cl.def("x2pdf", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::x2pdf, "C++: Pythia8::Info::x2pdf(int) const --> double", pybind11::arg("i"));
		cl.def("pdf1", [](Pythia8::Info const &o) -> double { return o.pdf1(); }, "");
		cl.def("pdf1", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::pdf1, "C++: Pythia8::Info::pdf1(int) const --> double", pybind11::arg("i"));
		cl.def("pdf2", [](Pythia8::Info const &o) -> double { return o.pdf2(); }, "");
		cl.def("pdf2", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::pdf2, "C++: Pythia8::Info::pdf2(int) const --> double", pybind11::arg("i"));
		cl.def("QFac", [](Pythia8::Info const &o) -> double { return o.QFac(); }, "");
		cl.def("QFac", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::QFac, "C++: Pythia8::Info::QFac(int) const --> double", pybind11::arg("i"));
		cl.def("Q2Fac", [](Pythia8::Info const &o) -> double { return o.Q2Fac(); }, "");
		cl.def("Q2Fac", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::Q2Fac, "C++: Pythia8::Info::Q2Fac(int) const --> double", pybind11::arg("i"));
		cl.def("isValence1", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isValence1, "C++: Pythia8::Info::isValence1() const --> bool");
		cl.def("isValence2", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isValence2, "C++: Pythia8::Info::isValence2() const --> bool");
		cl.def("alphaS", [](Pythia8::Info const &o) -> double { return o.alphaS(); }, "");
		cl.def("alphaS", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::alphaS, "C++: Pythia8::Info::alphaS(int) const --> double", pybind11::arg("i"));
		cl.def("alphaEM", [](Pythia8::Info const &o) -> double { return o.alphaEM(); }, "");
		cl.def("alphaEM", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::alphaEM, "C++: Pythia8::Info::alphaEM(int) const --> double", pybind11::arg("i"));
		cl.def("QRen", [](Pythia8::Info const &o) -> double { return o.QRen(); }, "");
		cl.def("QRen", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::QRen, "C++: Pythia8::Info::QRen(int) const --> double", pybind11::arg("i"));
		cl.def("Q2Ren", [](Pythia8::Info const &o) -> double { return o.Q2Ren(); }, "");
		cl.def("Q2Ren", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::Q2Ren, "C++: Pythia8::Info::Q2Ren(int) const --> double", pybind11::arg("i"));
		cl.def("scalup", [](Pythia8::Info const &o) -> double { return o.scalup(); }, "");
		cl.def("scalup", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::scalup, "C++: Pythia8::Info::scalup(int) const --> double", pybind11::arg("i"));
		cl.def("xGammaA", (double (Pythia8::Info::*)() const) &Pythia8::Info::xGammaA, "C++: Pythia8::Info::xGammaA() const --> double");
		cl.def("xGammaB", (double (Pythia8::Info::*)() const) &Pythia8::Info::xGammaB, "C++: Pythia8::Info::xGammaB() const --> double");
		cl.def("Q2GammaA", (double (Pythia8::Info::*)() const) &Pythia8::Info::Q2GammaA, "C++: Pythia8::Info::Q2GammaA() const --> double");
		cl.def("Q2GammaB", (double (Pythia8::Info::*)() const) &Pythia8::Info::Q2GammaB, "C++: Pythia8::Info::Q2GammaB() const --> double");
		cl.def("eCMsub", (double (Pythia8::Info::*)() const) &Pythia8::Info::eCMsub, "C++: Pythia8::Info::eCMsub() const --> double");
		cl.def("thetaScatLepA", (double (Pythia8::Info::*)() const) &Pythia8::Info::thetaScatLepA, "C++: Pythia8::Info::thetaScatLepA() const --> double");
		cl.def("thetaScatLepB", (double (Pythia8::Info::*)() const) &Pythia8::Info::thetaScatLepB, "C++: Pythia8::Info::thetaScatLepB() const --> double");
		cl.def("sHatNew", (double (Pythia8::Info::*)() const) &Pythia8::Info::sHatNew, "C++: Pythia8::Info::sHatNew() const --> double");
		cl.def("photonMode", (int (Pythia8::Info::*)() const) &Pythia8::Info::photonMode, "C++: Pythia8::Info::photonMode() const --> int");
		cl.def("isVMDstateA", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isVMDstateA, "C++: Pythia8::Info::isVMDstateA() const --> bool");
		cl.def("isVMDstateB", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isVMDstateB, "C++: Pythia8::Info::isVMDstateB() const --> bool");
		cl.def("idVMDA", (int (Pythia8::Info::*)() const) &Pythia8::Info::idVMDA, "C++: Pythia8::Info::idVMDA() const --> int");
		cl.def("idVMDB", (int (Pythia8::Info::*)() const) &Pythia8::Info::idVMDB, "C++: Pythia8::Info::idVMDB() const --> int");
		cl.def("mVMDA", (double (Pythia8::Info::*)() const) &Pythia8::Info::mVMDA, "C++: Pythia8::Info::mVMDA() const --> double");
		cl.def("mVMDB", (double (Pythia8::Info::*)() const) &Pythia8::Info::mVMDB, "C++: Pythia8::Info::mVMDB() const --> double");
		cl.def("scaleVMDA", (double (Pythia8::Info::*)() const) &Pythia8::Info::scaleVMDA, "C++: Pythia8::Info::scaleVMDA() const --> double");
		cl.def("scaleVMDB", (double (Pythia8::Info::*)() const) &Pythia8::Info::scaleVMDB, "C++: Pythia8::Info::scaleVMDB() const --> double");
		cl.def("mHat", [](Pythia8::Info const &o) -> double { return o.mHat(); }, "");
		cl.def("mHat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::mHat, "C++: Pythia8::Info::mHat(int) const --> double", pybind11::arg("i"));
		cl.def("sHat", [](Pythia8::Info const &o) -> double { return o.sHat(); }, "");
		cl.def("sHat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::sHat, "C++: Pythia8::Info::sHat(int) const --> double", pybind11::arg("i"));
		cl.def("tHat", [](Pythia8::Info const &o) -> double { return o.tHat(); }, "");
		cl.def("tHat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::tHat, "C++: Pythia8::Info::tHat(int) const --> double", pybind11::arg("i"));
		cl.def("uHat", [](Pythia8::Info const &o) -> double { return o.uHat(); }, "");
		cl.def("uHat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::uHat, "C++: Pythia8::Info::uHat(int) const --> double", pybind11::arg("i"));
		cl.def("pTHat", [](Pythia8::Info const &o) -> double { return o.pTHat(); }, "");
		cl.def("pTHat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::pTHat, "C++: Pythia8::Info::pTHat(int) const --> double", pybind11::arg("i"));
		cl.def("pT2Hat", [](Pythia8::Info const &o) -> double { return o.pT2Hat(); }, "");
		cl.def("pT2Hat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::pT2Hat, "C++: Pythia8::Info::pT2Hat(int) const --> double", pybind11::arg("i"));
		cl.def("m3Hat", [](Pythia8::Info const &o) -> double { return o.m3Hat(); }, "");
		cl.def("m3Hat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::m3Hat, "C++: Pythia8::Info::m3Hat(int) const --> double", pybind11::arg("i"));
		cl.def("m4Hat", [](Pythia8::Info const &o) -> double { return o.m4Hat(); }, "");
		cl.def("m4Hat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::m4Hat, "C++: Pythia8::Info::m4Hat(int) const --> double", pybind11::arg("i"));
		cl.def("thetaHat", [](Pythia8::Info const &o) -> double { return o.thetaHat(); }, "");
		cl.def("thetaHat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::thetaHat, "C++: Pythia8::Info::thetaHat(int) const --> double", pybind11::arg("i"));
		cl.def("phiHat", [](Pythia8::Info const &o) -> double { return o.phiHat(); }, "");
		cl.def("phiHat", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::phiHat, "C++: Pythia8::Info::phiHat(int) const --> double", pybind11::arg("i"));
		cl.def("weight", [](Pythia8::Info const &o) -> double { return o.weight(); }, "");
		cl.def("weight", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::weight, "C++: Pythia8::Info::weight(int) const --> double", pybind11::arg("i"));
		cl.def("weightSum", (double (Pythia8::Info::*)() const) &Pythia8::Info::weightSum, "C++: Pythia8::Info::weightSum() const --> double");
		cl.def("lhaStrategy", (double (Pythia8::Info::*)() const) &Pythia8::Info::lhaStrategy, "C++: Pythia8::Info::lhaStrategy() const --> double");
		cl.def("nWeights", (int (Pythia8::Info::*)() const) &Pythia8::Info::nWeights, "C++: Pythia8::Info::nWeights() const --> int");
		cl.def("weightLabel", (std::string (Pythia8::Info::*)(int) const) &Pythia8::Info::weightLabel, "C++: Pythia8::Info::weightLabel(int) const --> std::string", pybind11::arg("iWeight"));
		cl.def("nWeightGroups", (int (Pythia8::Info::*)() const) &Pythia8::Info::nWeightGroups, "C++: Pythia8::Info::nWeightGroups() const --> int");
		cl.def("getGroupName", (std::string (Pythia8::Info::*)(int) const) &Pythia8::Info::getGroupName, "C++: Pythia8::Info::getGroupName(int) const --> std::string", pybind11::arg("iGN"));
		cl.def("getGroupWeight", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::getGroupWeight, "C++: Pythia8::Info::getGroupWeight(int) const --> double", pybind11::arg("iGW"));
		cl.def("nISR", (int (Pythia8::Info::*)() const) &Pythia8::Info::nISR, "C++: Pythia8::Info::nISR() const --> int");
		cl.def("nFSRinProc", (int (Pythia8::Info::*)() const) &Pythia8::Info::nFSRinProc, "C++: Pythia8::Info::nFSRinProc() const --> int");
		cl.def("nFSRinRes", (int (Pythia8::Info::*)() const) &Pythia8::Info::nFSRinRes, "C++: Pythia8::Info::nFSRinRes() const --> int");
		cl.def("pTmaxMPI", (double (Pythia8::Info::*)() const) &Pythia8::Info::pTmaxMPI, "C++: Pythia8::Info::pTmaxMPI() const --> double");
		cl.def("pTmaxISR", (double (Pythia8::Info::*)() const) &Pythia8::Info::pTmaxISR, "C++: Pythia8::Info::pTmaxISR() const --> double");
		cl.def("pTmaxFSR", (double (Pythia8::Info::*)() const) &Pythia8::Info::pTmaxFSR, "C++: Pythia8::Info::pTmaxFSR() const --> double");
		cl.def("pTnow", (double (Pythia8::Info::*)() const) &Pythia8::Info::pTnow, "C++: Pythia8::Info::pTnow() const --> double");
		cl.def("a0MPI", (double (Pythia8::Info::*)() const) &Pythia8::Info::a0MPI, "C++: Pythia8::Info::a0MPI() const --> double");
		cl.def("bMPI", (double (Pythia8::Info::*)() const) &Pythia8::Info::bMPI, "C++: Pythia8::Info::bMPI() const --> double");
		cl.def("enhanceMPI", (double (Pythia8::Info::*)() const) &Pythia8::Info::enhanceMPI, "C++: Pythia8::Info::enhanceMPI() const --> double");
		cl.def("enhanceMPIavg", (double (Pythia8::Info::*)() const) &Pythia8::Info::enhanceMPIavg, "C++: Pythia8::Info::enhanceMPIavg() const --> double");
		cl.def("eMPI", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::eMPI, "C++: Pythia8::Info::eMPI(int) const --> double", pybind11::arg("i"));
		cl.def("bMPIold", (double (Pythia8::Info::*)() const) &Pythia8::Info::bMPIold, "C++: Pythia8::Info::bMPIold() const --> double");
		cl.def("enhanceMPIold", (double (Pythia8::Info::*)() const) &Pythia8::Info::enhanceMPIold, "C++: Pythia8::Info::enhanceMPIold() const --> double");
		cl.def("enhanceMPIoldavg", (double (Pythia8::Info::*)() const) &Pythia8::Info::enhanceMPIoldavg, "C++: Pythia8::Info::enhanceMPIoldavg() const --> double");
		cl.def("nMPI", (int (Pythia8::Info::*)() const) &Pythia8::Info::nMPI, "C++: Pythia8::Info::nMPI() const --> int");
		cl.def("codeMPI", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::codeMPI, "C++: Pythia8::Info::codeMPI(int) const --> int", pybind11::arg("i"));
		cl.def("pTMPI", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::pTMPI, "C++: Pythia8::Info::pTMPI(int) const --> double", pybind11::arg("i"));
		cl.def("iAMPI", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::iAMPI, "C++: Pythia8::Info::iAMPI(int) const --> int", pybind11::arg("i"));
		cl.def("iBMPI", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::iBMPI, "C++: Pythia8::Info::iBMPI(int) const --> int", pybind11::arg("i"));
		cl.def("codesHard", (class std::vector<int, class std::allocator<int> > (Pythia8::Info::*)()) &Pythia8::Info::codesHard, "C++: Pythia8::Info::codesHard() --> class std::vector<int, class std::allocator<int> >");
		cl.def("nameProc", [](Pythia8::Info const &o) -> std::string { return o.nameProc(); }, "");
		cl.def("nameProc", (std::string (Pythia8::Info::*)(int) const) &Pythia8::Info::nameProc, "C++: Pythia8::Info::nameProc(int) const --> std::string", pybind11::arg("i"));
		cl.def("nTried", [](Pythia8::Info const &o) -> long { return o.nTried(); }, "");
		cl.def("nTried", (long (Pythia8::Info::*)(int) const) &Pythia8::Info::nTried, "C++: Pythia8::Info::nTried(int) const --> long", pybind11::arg("i"));
		cl.def("nSelected", [](Pythia8::Info const &o) -> long { return o.nSelected(); }, "");
		cl.def("nSelected", (long (Pythia8::Info::*)(int) const) &Pythia8::Info::nSelected, "C++: Pythia8::Info::nSelected(int) const --> long", pybind11::arg("i"));
		cl.def("nAccepted", [](Pythia8::Info const &o) -> long { return o.nAccepted(); }, "");
		cl.def("nAccepted", (long (Pythia8::Info::*)(int) const) &Pythia8::Info::nAccepted, "C++: Pythia8::Info::nAccepted(int) const --> long", pybind11::arg("i"));
		cl.def("sigmaGen", [](Pythia8::Info const &o) -> double { return o.sigmaGen(); }, "");
		cl.def("sigmaGen", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::sigmaGen, "C++: Pythia8::Info::sigmaGen(int) const --> double", pybind11::arg("i"));
		cl.def("sigmaErr", [](Pythia8::Info const &o) -> double { return o.sigmaErr(); }, "");
		cl.def("sigmaErr", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::sigmaErr, "C++: Pythia8::Info::sigmaErr(int) const --> double", pybind11::arg("i"));
		cl.def("getCounter", (int (Pythia8::Info::*)(int) const) &Pythia8::Info::getCounter, "C++: Pythia8::Info::getCounter(int) const --> int", pybind11::arg("i"));
		cl.def("setCounter", [](Pythia8::Info &o, int const & a0) -> void { return o.setCounter(a0); }, "", pybind11::arg("i"));
		cl.def("setCounter", (void (Pythia8::Info::*)(int, int)) &Pythia8::Info::setCounter, "C++: Pythia8::Info::setCounter(int, int) --> void", pybind11::arg("i"), pybind11::arg("value"));
		cl.def("addCounter", [](Pythia8::Info &o, int const & a0) -> void { return o.addCounter(a0); }, "", pybind11::arg("i"));
		cl.def("addCounter", (void (Pythia8::Info::*)(int, int)) &Pythia8::Info::addCounter, "C++: Pythia8::Info::addCounter(int, int) --> void", pybind11::arg("i"), pybind11::arg("value"));
		cl.def("errorReset", (void (Pythia8::Info::*)()) &Pythia8::Info::errorReset, "C++: Pythia8::Info::errorReset() --> void");
		cl.def("errorMsg", [](Pythia8::Info &o, class std::basic_string<char> const & a0) -> void { return o.errorMsg(a0); }, "", pybind11::arg("messageIn"));
		cl.def("errorMsg", [](Pythia8::Info &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1) -> void { return o.errorMsg(a0, a1); }, "", pybind11::arg("messageIn"), pybind11::arg("extraIn"));
		cl.def("errorMsg", (void (Pythia8::Info::*)(std::string, std::string, bool)) &Pythia8::Info::errorMsg, "C++: Pythia8::Info::errorMsg(std::string, std::string, bool) --> void", pybind11::arg("messageIn"), pybind11::arg("extraIn"), pybind11::arg("showAlways"));
		cl.def("errorTotalNumber", (int (Pythia8::Info::*)() const) &Pythia8::Info::errorTotalNumber, "C++: Pythia8::Info::errorTotalNumber() const --> int");
		cl.def("errorStatistics", (void (Pythia8::Info::*)() const) &Pythia8::Info::errorStatistics, "C++: Pythia8::Info::errorStatistics() const --> void");
		cl.def("setTooLowPTmin", (void (Pythia8::Info::*)(bool)) &Pythia8::Info::setTooLowPTmin, "C++: Pythia8::Info::setTooLowPTmin(bool) --> void", pybind11::arg("lowPTminIn"));
		cl.def("setValence", (void (Pythia8::Info::*)(bool, bool)) &Pythia8::Info::setValence, "C++: Pythia8::Info::setValence(bool, bool) --> void", pybind11::arg("isVal1In"), pybind11::arg("isVal2In"));
		cl.def("hasHistory", (void (Pythia8::Info::*)(bool)) &Pythia8::Info::hasHistory, "C++: Pythia8::Info::hasHistory(bool) --> void", pybind11::arg("hasHistoryIn"));
		cl.def("hasHistory", (bool (Pythia8::Info::*)()) &Pythia8::Info::hasHistory, "C++: Pythia8::Info::hasHistory() --> bool");
		cl.def("zNowISR", (void (Pythia8::Info::*)(double)) &Pythia8::Info::zNowISR, "C++: Pythia8::Info::zNowISR(double) --> void", pybind11::arg("zNowIn"));
		cl.def("zNowISR", (double (Pythia8::Info::*)()) &Pythia8::Info::zNowISR, "C++: Pythia8::Info::zNowISR() --> double");
		cl.def("pT2NowISR", (void (Pythia8::Info::*)(double)) &Pythia8::Info::pT2NowISR, "C++: Pythia8::Info::pT2NowISR(double) --> void", pybind11::arg("pT2NowIn"));
		cl.def("pT2NowISR", (double (Pythia8::Info::*)()) &Pythia8::Info::pT2NowISR, "C++: Pythia8::Info::pT2NowISR() --> double");
		cl.def("mergingWeight", [](Pythia8::Info const &o) -> double { return o.mergingWeight(); }, "");
		cl.def("mergingWeight", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::mergingWeight, "C++: Pythia8::Info::mergingWeight(int) const --> double", pybind11::arg("i"));
		cl.def("mergingWeightNLO", [](Pythia8::Info const &o) -> double { return o.mergingWeightNLO(); }, "");
		cl.def("mergingWeightNLO", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::mergingWeightNLO, "C++: Pythia8::Info::mergingWeightNLO(int) const --> double", pybind11::arg("i"));
		cl.def("header", (std::string (Pythia8::Info::*)(const std::string &) const) &Pythia8::Info::header, "C++: Pythia8::Info::header(const std::string &) const --> std::string", pybind11::arg("key"));
		cl.def("headerKeys", (class std::vector<std::string, class std::allocator<std::string > > (Pythia8::Info::*)() const) &Pythia8::Info::headerKeys, "C++: Pythia8::Info::headerKeys() const --> class std::vector<std::string, class std::allocator<std::string > >");
		cl.def("nProcessesLHEF", (int (Pythia8::Info::*)() const) &Pythia8::Info::nProcessesLHEF, "C++: Pythia8::Info::nProcessesLHEF() const --> int");
		cl.def("sigmaLHEF", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::sigmaLHEF, "C++: Pythia8::Info::sigmaLHEF(int) const --> double", pybind11::arg("iProcess"));
		cl.def("setLHEF3InitInfo", (void (Pythia8::Info::*)()) &Pythia8::Info::setLHEF3InitInfo, "C++: Pythia8::Info::setLHEF3InitInfo() --> void");
		cl.def("setLHEF3EventInfo", (void (Pythia8::Info::*)()) &Pythia8::Info::setLHEF3EventInfo, "C++: Pythia8::Info::setLHEF3EventInfo() --> void");
		cl.def("getEventAttribute", [](Pythia8::Info const &o, class std::basic_string<char> const & a0) -> std::string { return o.getEventAttribute(a0); }, "", pybind11::arg("key"));
		cl.def("getEventAttribute", (std::string (Pythia8::Info::*)(std::string, bool) const) &Pythia8::Info::getEventAttribute, "C++: Pythia8::Info::getEventAttribute(std::string, bool) const --> std::string", pybind11::arg("key"), pybind11::arg("doRemoveWhitespace"));
		cl.def("setEventAttribute", [](Pythia8::Info &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1) -> void { return o.setEventAttribute(a0, a1); }, "", pybind11::arg("key"), pybind11::arg("value"));
		cl.def("setEventAttribute", (void (Pythia8::Info::*)(std::string, std::string, bool)) &Pythia8::Info::setEventAttribute, "C++: Pythia8::Info::setEventAttribute(std::string, std::string, bool) --> void", pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("doOverwrite"));
		cl.def("LHEFversion", (int (Pythia8::Info::*)() const) &Pythia8::Info::LHEFversion, "C++: Pythia8::Info::LHEFversion() const --> int");
		cl.def("getInitrwgtSize", (unsigned int (Pythia8::Info::*)() const) &Pythia8::Info::getInitrwgtSize, "C++: Pythia8::Info::getInitrwgtSize() const --> unsigned int");
		cl.def("getGeneratorSize", (unsigned int (Pythia8::Info::*)() const) &Pythia8::Info::getGeneratorSize, "C++: Pythia8::Info::getGeneratorSize() const --> unsigned int");
		cl.def("getGeneratorValue", [](Pythia8::Info const &o) -> std::string { return o.getGeneratorValue(); }, "");
		cl.def("getGeneratorValue", (std::string (Pythia8::Info::*)(unsigned int) const) &Pythia8::Info::getGeneratorValue, "C++: Pythia8::Info::getGeneratorValue(unsigned int) const --> std::string", pybind11::arg("n"));
		cl.def("getGeneratorAttribute", [](Pythia8::Info const &o, unsigned int const & a0, class std::basic_string<char> const & a1) -> std::string { return o.getGeneratorAttribute(a0, a1); }, "", pybind11::arg("n"), pybind11::arg("key"));
		cl.def("getGeneratorAttribute", (std::string (Pythia8::Info::*)(unsigned int, std::string, bool) const) &Pythia8::Info::getGeneratorAttribute, "C++: Pythia8::Info::getGeneratorAttribute(unsigned int, std::string, bool) const --> std::string", pybind11::arg("n"), pybind11::arg("key"), pybind11::arg("doRemoveWhitespace"));
		cl.def("getWeightsDetailedSize", (unsigned int (Pythia8::Info::*)() const) &Pythia8::Info::getWeightsDetailedSize, "C++: Pythia8::Info::getWeightsDetailedSize() const --> unsigned int");
		cl.def("getWeightsDetailedValue", (double (Pythia8::Info::*)(std::string) const) &Pythia8::Info::getWeightsDetailedValue, "C++: Pythia8::Info::getWeightsDetailedValue(std::string) const --> double", pybind11::arg("n"));
		cl.def("getWeightsDetailedAttribute", [](Pythia8::Info const &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1) -> std::string { return o.getWeightsDetailedAttribute(a0, a1); }, "", pybind11::arg("n"), pybind11::arg("key"));
		cl.def("getWeightsDetailedAttribute", (std::string (Pythia8::Info::*)(std::string, std::string, bool) const) &Pythia8::Info::getWeightsDetailedAttribute, "C++: Pythia8::Info::getWeightsDetailedAttribute(std::string, std::string, bool) const --> std::string", pybind11::arg("n"), pybind11::arg("key"), pybind11::arg("doRemoveWhitespace"));
		cl.def("getWeightsCompressedSize", (unsigned int (Pythia8::Info::*)() const) &Pythia8::Info::getWeightsCompressedSize, "C++: Pythia8::Info::getWeightsCompressedSize() const --> unsigned int");
		cl.def("getWeightsCompressedValue", (double (Pythia8::Info::*)(unsigned int) const) &Pythia8::Info::getWeightsCompressedValue, "C++: Pythia8::Info::getWeightsCompressedValue(unsigned int) const --> double", pybind11::arg("n"));
		cl.def("getWeightsCompressedAttribute", [](Pythia8::Info const &o, class std::basic_string<char> const & a0) -> std::string { return o.getWeightsCompressedAttribute(a0); }, "", pybind11::arg("key"));
		cl.def("getWeightsCompressedAttribute", (std::string (Pythia8::Info::*)(std::string, bool) const) &Pythia8::Info::getWeightsCompressedAttribute, "C++: Pythia8::Info::getWeightsCompressedAttribute(std::string, bool) const --> std::string", pybind11::arg("key"), pybind11::arg("doRemoveWhitespace"));
		cl.def("getScalesValue", [](Pythia8::Info const &o) -> std::string { return o.getScalesValue(); }, "");
		cl.def("getScalesValue", (std::string (Pythia8::Info::*)(bool) const) &Pythia8::Info::getScalesValue, "C++: Pythia8::Info::getScalesValue(bool) const --> std::string", pybind11::arg("doRemoveWhitespace"));
		cl.def("getScalesAttribute", (double (Pythia8::Info::*)(std::string) const) &Pythia8::Info::getScalesAttribute, "C++: Pythia8::Info::getScalesAttribute(std::string) const --> double", pybind11::arg("key"));
		cl.def("getHeaderBlock", (std::string (Pythia8::Info::*)() const) &Pythia8::Info::getHeaderBlock, "C++: Pythia8::Info::getHeaderBlock() const --> std::string");
		cl.def("getEventComments", (std::string (Pythia8::Info::*)() const) &Pythia8::Info::getEventComments, "C++: Pythia8::Info::getEventComments() const --> std::string");
		cl.def("setHeader", (void (Pythia8::Info::*)(const std::string &, const std::string &)) &Pythia8::Info::setHeader, "C++: Pythia8::Info::setHeader(const std::string &, const std::string &) --> void", pybind11::arg("key"), pybind11::arg("val"));
		cl.def("setAbortPartonLevel", (void (Pythia8::Info::*)(bool)) &Pythia8::Info::setAbortPartonLevel, "C++: Pythia8::Info::setAbortPartonLevel(bool) --> void", pybind11::arg("abortIn"));
		cl.def("getAbortPartonLevel", (bool (Pythia8::Info::*)() const) &Pythia8::Info::getAbortPartonLevel, "C++: Pythia8::Info::getAbortPartonLevel() const --> bool");
		cl.def("hasUnresolvedBeams", (bool (Pythia8::Info::*)() const) &Pythia8::Info::hasUnresolvedBeams, "C++: Pythia8::Info::hasUnresolvedBeams() const --> bool");
		cl.def("hasPomPsystem", (bool (Pythia8::Info::*)() const) &Pythia8::Info::hasPomPsystem, "C++: Pythia8::Info::hasPomPsystem() const --> bool");
		cl.def("isHardDiffractive", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isHardDiffractive, "C++: Pythia8::Info::isHardDiffractive() const --> bool");
		cl.def("isHardDiffractiveA", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isHardDiffractiveA, "C++: Pythia8::Info::isHardDiffractiveA() const --> bool");
		cl.def("isHardDiffractiveB", (bool (Pythia8::Info::*)() const) &Pythia8::Info::isHardDiffractiveB, "C++: Pythia8::Info::isHardDiffractiveB() const --> bool");
		cl.def("xPomeronA", (double (Pythia8::Info::*)() const) &Pythia8::Info::xPomeronA, "C++: Pythia8::Info::xPomeronA() const --> double");
		cl.def("xPomeronB", (double (Pythia8::Info::*)() const) &Pythia8::Info::xPomeronB, "C++: Pythia8::Info::xPomeronB() const --> double");
		cl.def("tPomeronA", (double (Pythia8::Info::*)() const) &Pythia8::Info::tPomeronA, "C++: Pythia8::Info::tPomeronA() const --> double");
		cl.def("tPomeronB", (double (Pythia8::Info::*)() const) &Pythia8::Info::tPomeronB, "C++: Pythia8::Info::tPomeronB() const --> double");
		cl.def("getWeakModes", (class std::vector<int, class std::allocator<int> > (Pythia8::Info::*)() const) &Pythia8::Info::getWeakModes, "C++: Pythia8::Info::getWeakModes() const --> class std::vector<int, class std::allocator<int> >");
		cl.def("getWeakDipoles", (class std::vector<struct std::pair<int, int>, class std::allocator<struct std::pair<int, int> > > (Pythia8::Info::*)() const) &Pythia8::Info::getWeakDipoles, "C++: Pythia8::Info::getWeakDipoles() const --> class std::vector<struct std::pair<int, int>, class std::allocator<struct std::pair<int, int> > >");
		cl.def("getWeakMomenta", (class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> > (Pythia8::Info::*)() const) &Pythia8::Info::getWeakMomenta, "C++: Pythia8::Info::getWeakMomenta() const --> class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> >");
		cl.def("getWeak2to2lines", (class std::vector<int, class std::allocator<int> > (Pythia8::Info::*)() const) &Pythia8::Info::getWeak2to2lines, "C++: Pythia8::Info::getWeak2to2lines() const --> class std::vector<int, class std::allocator<int> >");
		cl.def("setWeakModes", (void (Pythia8::Info::*)(class std::vector<int, class std::allocator<int> >)) &Pythia8::Info::setWeakModes, "C++: Pythia8::Info::setWeakModes(class std::vector<int, class std::allocator<int> >) --> void", pybind11::arg("weakModesIn"));
		cl.def("setWeakDipoles", (void (Pythia8::Info::*)(class std::vector<struct std::pair<int, int>, class std::allocator<struct std::pair<int, int> > >)) &Pythia8::Info::setWeakDipoles, "C++: Pythia8::Info::setWeakDipoles(class std::vector<struct std::pair<int, int>, class std::allocator<struct std::pair<int, int> > >) --> void", pybind11::arg("weakDipolesIn"));
		cl.def("setWeakMomenta", (void (Pythia8::Info::*)(class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> >)) &Pythia8::Info::setWeakMomenta, "C++: Pythia8::Info::setWeakMomenta(class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> >) --> void", pybind11::arg("weakMomentaIn"));
		cl.def("setWeak2to2lines", (void (Pythia8::Info::*)(class std::vector<int, class std::allocator<int> >)) &Pythia8::Info::setWeak2to2lines, "C++: Pythia8::Info::setWeak2to2lines(class std::vector<int, class std::allocator<int> >) --> void", pybind11::arg("weak2to2linesIn"));
		cl.def("plugin", (class std::shared_ptr<class Pythia8::Plugin> (Pythia8::Info::*)(std::string)) &Pythia8::Info::plugin, "C++: Pythia8::Info::plugin(std::string) --> class std::shared_ptr<class Pythia8::Plugin>", pybind11::arg("nameIn"));
		cl.def("setBeamA", (void (Pythia8::Info::*)(int, double, double, double)) &Pythia8::Info::setBeamA, "C++: Pythia8::Info::setBeamA(int, double, double, double) --> void", pybind11::arg("idAin"), pybind11::arg("pzAin"), pybind11::arg("eAin"), pybind11::arg("mAin"));
		cl.def("setBeamB", (void (Pythia8::Info::*)(int, double, double, double)) &Pythia8::Info::setBeamB, "C++: Pythia8::Info::setBeamB(int, double, double, double) --> void", pybind11::arg("idBin"), pybind11::arg("pzBin"), pybind11::arg("eBin"), pybind11::arg("mBin"));
		cl.def("setECM", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setECM, "C++: Pythia8::Info::setECM(double) --> void", pybind11::arg("eCMin"));
		cl.def("setX1Gamma", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setX1Gamma, "C++: Pythia8::Info::setX1Gamma(double) --> void", pybind11::arg("x1GammaIn"));
		cl.def("setX2Gamma", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setX2Gamma, "C++: Pythia8::Info::setX2Gamma(double) --> void", pybind11::arg("x2GammaIn"));
		cl.def("setQ2Gamma1", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setQ2Gamma1, "C++: Pythia8::Info::setQ2Gamma1(double) --> void", pybind11::arg("Q2gammaIn"));
		cl.def("setQ2Gamma2", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setQ2Gamma2, "C++: Pythia8::Info::setQ2Gamma2(double) --> void", pybind11::arg("Q2gammaIn"));
		cl.def("setTheta1", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setTheta1, "C++: Pythia8::Info::setTheta1(double) --> void", pybind11::arg("theta1In"));
		cl.def("setTheta2", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setTheta2, "C++: Pythia8::Info::setTheta2(double) --> void", pybind11::arg("theta2In"));
		cl.def("setECMsub", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setECMsub, "C++: Pythia8::Info::setECMsub(double) --> void", pybind11::arg("eCMsubIn"));
		cl.def("setsHatNew", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setsHatNew, "C++: Pythia8::Info::setsHatNew(double) --> void", pybind11::arg("sHatNewIn"));
		cl.def("setGammaMode", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setGammaMode, "C++: Pythia8::Info::setGammaMode(double) --> void", pybind11::arg("gammaModeIn"));
		cl.def("setVMDstateA", (void (Pythia8::Info::*)(bool, int, double, double)) &Pythia8::Info::setVMDstateA, "C++: Pythia8::Info::setVMDstateA(bool, int, double, double) --> void", pybind11::arg("isVMDAIn"), pybind11::arg("idAIn"), pybind11::arg("mAIn"), pybind11::arg("scaleAIn"));
		cl.def("setVMDstateB", (void (Pythia8::Info::*)(bool, int, double, double)) &Pythia8::Info::setVMDstateB, "C++: Pythia8::Info::setVMDstateB(bool, int, double, double) --> void", pybind11::arg("isVMDBIn"), pybind11::arg("idBIn"), pybind11::arg("mBIn"), pybind11::arg("scaleBIn"));
		cl.def("clear", (void (Pythia8::Info::*)()) &Pythia8::Info::clear, "C++: Pythia8::Info::clear() --> void");
		cl.def("sizeMPIarrays", (int (Pythia8::Info::*)() const) &Pythia8::Info::sizeMPIarrays, "C++: Pythia8::Info::sizeMPIarrays() const --> int");
		cl.def("resizeMPIarrays", (void (Pythia8::Info::*)(int)) &Pythia8::Info::resizeMPIarrays, "C++: Pythia8::Info::resizeMPIarrays(int) --> void", pybind11::arg("newSize"));
		cl.def("setType", [](Pythia8::Info &o, class std::basic_string<char> const & a0, int const & a1, int const & a2) -> void { return o.setType(a0, a1, a2); }, "", pybind11::arg("nameIn"), pybind11::arg("codeIn"), pybind11::arg("nFinalIn"));
		cl.def("setType", [](Pythia8::Info &o, class std::basic_string<char> const & a0, int const & a1, int const & a2, bool const & a3) -> void { return o.setType(a0, a1, a2, a3); }, "", pybind11::arg("nameIn"), pybind11::arg("codeIn"), pybind11::arg("nFinalIn"), pybind11::arg("isNonDiffIn"));
		cl.def("setType", [](Pythia8::Info &o, class std::basic_string<char> const & a0, int const & a1, int const & a2, bool const & a3, bool const & a4) -> void { return o.setType(a0, a1, a2, a3, a4); }, "", pybind11::arg("nameIn"), pybind11::arg("codeIn"), pybind11::arg("nFinalIn"), pybind11::arg("isNonDiffIn"), pybind11::arg("isResolvedIn"));
		cl.def("setType", [](Pythia8::Info &o, class std::basic_string<char> const & a0, int const & a1, int const & a2, bool const & a3, bool const & a4, bool const & a5) -> void { return o.setType(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("nameIn"), pybind11::arg("codeIn"), pybind11::arg("nFinalIn"), pybind11::arg("isNonDiffIn"), pybind11::arg("isResolvedIn"), pybind11::arg("isDiffractiveAin"));
		cl.def("setType", [](Pythia8::Info &o, class std::basic_string<char> const & a0, int const & a1, int const & a2, bool const & a3, bool const & a4, bool const & a5, bool const & a6) -> void { return o.setType(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("nameIn"), pybind11::arg("codeIn"), pybind11::arg("nFinalIn"), pybind11::arg("isNonDiffIn"), pybind11::arg("isResolvedIn"), pybind11::arg("isDiffractiveAin"), pybind11::arg("isDiffractiveBin"));
		cl.def("setType", [](Pythia8::Info &o, class std::basic_string<char> const & a0, int const & a1, int const & a2, bool const & a3, bool const & a4, bool const & a5, bool const & a6, bool const & a7) -> void { return o.setType(a0, a1, a2, a3, a4, a5, a6, a7); }, "", pybind11::arg("nameIn"), pybind11::arg("codeIn"), pybind11::arg("nFinalIn"), pybind11::arg("isNonDiffIn"), pybind11::arg("isResolvedIn"), pybind11::arg("isDiffractiveAin"), pybind11::arg("isDiffractiveBin"), pybind11::arg("isDiffractiveCin"));
		cl.def("setType", (void (Pythia8::Info::*)(std::string, int, int, bool, bool, bool, bool, bool, bool)) &Pythia8::Info::setType, "C++: Pythia8::Info::setType(std::string, int, int, bool, bool, bool, bool, bool, bool) --> void", pybind11::arg("nameIn"), pybind11::arg("codeIn"), pybind11::arg("nFinalIn"), pybind11::arg("isNonDiffIn"), pybind11::arg("isResolvedIn"), pybind11::arg("isDiffractiveAin"), pybind11::arg("isDiffractiveBin"), pybind11::arg("isDiffractiveCin"), pybind11::arg("isLHAin"));
		cl.def("setSubType", (void (Pythia8::Info::*)(int, std::string, int, int)) &Pythia8::Info::setSubType, "C++: Pythia8::Info::setSubType(int, std::string, int, int) --> void", pybind11::arg("iDS"), pybind11::arg("nameSubIn"), pybind11::arg("codeSubIn"), pybind11::arg("nFinalSubIn"));
		cl.def("setPDFalpha", (void (Pythia8::Info::*)(int, int, int, double, double, double, double, double, double, double, double, double)) &Pythia8::Info::setPDFalpha, "C++: Pythia8::Info::setPDFalpha(int, int, int, double, double, double, double, double, double, double, double, double) --> void", pybind11::arg("iDS"), pybind11::arg("id1pdfIn"), pybind11::arg("id2pdfIn"), pybind11::arg("x1pdfIn"), pybind11::arg("x2pdfIn"), pybind11::arg("pdf1In"), pybind11::arg("pdf2In"), pybind11::arg("Q2FacIn"), pybind11::arg("alphaEMIn"), pybind11::arg("alphaSIn"), pybind11::arg("Q2RenIn"), pybind11::arg("scalupIn"));
		cl.def("setScalup", (void (Pythia8::Info::*)(int, double)) &Pythia8::Info::setScalup, "C++: Pythia8::Info::setScalup(int, double) --> void", pybind11::arg("iDS"), pybind11::arg("scalupIn"));
		cl.def("setKin", (void (Pythia8::Info::*)(int, int, int, double, double, double, double, double, double, double, double, double, double)) &Pythia8::Info::setKin, "C++: Pythia8::Info::setKin(int, int, int, double, double, double, double, double, double, double, double, double, double) --> void", pybind11::arg("iDS"), pybind11::arg("id1In"), pybind11::arg("id2In"), pybind11::arg("x1In"), pybind11::arg("x2In"), pybind11::arg("sHatIn"), pybind11::arg("tHatIn"), pybind11::arg("uHatIn"), pybind11::arg("pTHatIn"), pybind11::arg("m3HatIn"), pybind11::arg("m4HatIn"), pybind11::arg("thetaHatIn"), pybind11::arg("phiHatIn"));
		cl.def("setTypeMPI", [](Pythia8::Info &o, int const & a0, double const & a1) -> void { return o.setTypeMPI(a0, a1); }, "", pybind11::arg("codeMPIIn"), pybind11::arg("pTMPIIn"));
		cl.def("setTypeMPI", [](Pythia8::Info &o, int const & a0, double const & a1, int const & a2) -> void { return o.setTypeMPI(a0, a1, a2); }, "", pybind11::arg("codeMPIIn"), pybind11::arg("pTMPIIn"), pybind11::arg("iAMPIIn"));
		cl.def("setTypeMPI", [](Pythia8::Info &o, int const & a0, double const & a1, int const & a2, int const & a3) -> void { return o.setTypeMPI(a0, a1, a2, a3); }, "", pybind11::arg("codeMPIIn"), pybind11::arg("pTMPIIn"), pybind11::arg("iAMPIIn"), pybind11::arg("iBMPIIn"));
		cl.def("setTypeMPI", (void (Pythia8::Info::*)(int, double, int, int, double)) &Pythia8::Info::setTypeMPI, "C++: Pythia8::Info::setTypeMPI(int, double, int, int, double) --> void", pybind11::arg("codeMPIIn"), pybind11::arg("pTMPIIn"), pybind11::arg("iAMPIIn"), pybind11::arg("iBMPIIn"), pybind11::arg("eMPIIn"));
		cl.def("sigmaReset", (void (Pythia8::Info::*)()) &Pythia8::Info::sigmaReset, "C++: Pythia8::Info::sigmaReset() --> void");
		cl.def("setSigma", (void (Pythia8::Info::*)(int, std::string, long, long, long, double, double, double)) &Pythia8::Info::setSigma, "C++: Pythia8::Info::setSigma(int, std::string, long, long, long, double, double, double) --> void", pybind11::arg("i"), pybind11::arg("procNameIn"), pybind11::arg("nTryIn"), pybind11::arg("nSelIn"), pybind11::arg("nAccIn"), pybind11::arg("sigGenIn"), pybind11::arg("sigErrIn"), pybind11::arg("wtAccSumIn"));
		cl.def("addSigma", (void (Pythia8::Info::*)(int, long, long, long, double, double)) &Pythia8::Info::addSigma, "C++: Pythia8::Info::addSigma(int, long, long, long, double, double) --> void", pybind11::arg("i"), pybind11::arg("nTryIn"), pybind11::arg("nSelIn"), pybind11::arg("nAccIn"), pybind11::arg("sigGenIn"), pybind11::arg("sigErrIn"));
		cl.def("setImpact", [](Pythia8::Info &o, double const & a0, double const & a1, double const & a2) -> void { return o.setImpact(a0, a1, a2); }, "", pybind11::arg("bMPIIn"), pybind11::arg("enhanceMPIIn"), pybind11::arg("enhanceMPIavgIn"));
		cl.def("setImpact", [](Pythia8::Info &o, double const & a0, double const & a1, double const & a2, bool const & a3) -> void { return o.setImpact(a0, a1, a2, a3); }, "", pybind11::arg("bMPIIn"), pybind11::arg("enhanceMPIIn"), pybind11::arg("enhanceMPIavgIn"), pybind11::arg("bIsSetIn"));
		cl.def("setImpact", (void (Pythia8::Info::*)(double, double, double, bool, bool)) &Pythia8::Info::setImpact, "C++: Pythia8::Info::setImpact(double, double, double, bool, bool) --> void", pybind11::arg("bMPIIn"), pybind11::arg("enhanceMPIIn"), pybind11::arg("enhanceMPIavgIn"), pybind11::arg("bIsSetIn"), pybind11::arg("pushBack"));
		cl.def("setPartEvolved", (void (Pythia8::Info::*)(int, int)) &Pythia8::Info::setPartEvolved, "C++: Pythia8::Info::setPartEvolved(int, int) --> void", pybind11::arg("nMPIIn"), pybind11::arg("nISRIn"));
		cl.def("setEvolution", (void (Pythia8::Info::*)(double, double, double, int, int, int, int)) &Pythia8::Info::setEvolution, "C++: Pythia8::Info::setEvolution(double, double, double, int, int, int, int) --> void", pybind11::arg("pTmaxMPIIn"), pybind11::arg("pTmaxISRIn"), pybind11::arg("pTmaxFSRIn"), pybind11::arg("nMPIIn"), pybind11::arg("nISRIn"), pybind11::arg("nFSRinProcIn"), pybind11::arg("nFSRinResIn"));
		cl.def("setPTnow", (void (Pythia8::Info::*)(double)) &Pythia8::Info::setPTnow, "C++: Pythia8::Info::setPTnow(double) --> void", pybind11::arg("pTnowIn"));
		cl.def("seta0MPI", (void (Pythia8::Info::*)(double)) &Pythia8::Info::seta0MPI, "C++: Pythia8::Info::seta0MPI(double) --> void", pybind11::arg("a0MPIin"));
		cl.def("setEndOfFile", (void (Pythia8::Info::*)(bool)) &Pythia8::Info::setEndOfFile, "C++: Pythia8::Info::setEndOfFile(bool) --> void", pybind11::arg("atEOFin"));
		cl.def("setWeight", (void (Pythia8::Info::*)(double, int)) &Pythia8::Info::setWeight, "C++: Pythia8::Info::setWeight(double, int) --> void", pybind11::arg("weightIn"), pybind11::arg("lhaStrategyIn"));
		cl.def("setIsResolved", (void (Pythia8::Info::*)(bool)) &Pythia8::Info::setIsResolved, "C++: Pythia8::Info::setIsResolved(bool) --> void", pybind11::arg("isResIn"));
		cl.def("setHardDiff", [](Pythia8::Info &o) -> void { return o.setHardDiff(); }, "");
		cl.def("setHardDiff", [](Pythia8::Info &o, bool const & a0) -> void { return o.setHardDiff(a0); }, "", pybind11::arg("hasUnresBeamsIn"));
		cl.def("setHardDiff", [](Pythia8::Info &o, bool const & a0, bool const & a1) -> void { return o.setHardDiff(a0, a1); }, "", pybind11::arg("hasUnresBeamsIn"), pybind11::arg("hasPomPsysIn"));
		cl.def("setHardDiff", [](Pythia8::Info &o, bool const & a0, bool const & a1, bool const & a2) -> void { return o.setHardDiff(a0, a1, a2); }, "", pybind11::arg("hasUnresBeamsIn"), pybind11::arg("hasPomPsysIn"), pybind11::arg("isHardDiffAIn"));
		cl.def("setHardDiff", [](Pythia8::Info &o, bool const & a0, bool const & a1, bool const & a2, bool const & a3) -> void { return o.setHardDiff(a0, a1, a2, a3); }, "", pybind11::arg("hasUnresBeamsIn"), pybind11::arg("hasPomPsysIn"), pybind11::arg("isHardDiffAIn"), pybind11::arg("isHardDiffBIn"));
		cl.def("setHardDiff", [](Pythia8::Info &o, bool const & a0, bool const & a1, bool const & a2, bool const & a3, double const & a4) -> void { return o.setHardDiff(a0, a1, a2, a3, a4); }, "", pybind11::arg("hasUnresBeamsIn"), pybind11::arg("hasPomPsysIn"), pybind11::arg("isHardDiffAIn"), pybind11::arg("isHardDiffBIn"), pybind11::arg("xPomAIn"));
		cl.def("setHardDiff", [](Pythia8::Info &o, bool const & a0, bool const & a1, bool const & a2, bool const & a3, double const & a4, double const & a5) -> void { return o.setHardDiff(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("hasUnresBeamsIn"), pybind11::arg("hasPomPsysIn"), pybind11::arg("isHardDiffAIn"), pybind11::arg("isHardDiffBIn"), pybind11::arg("xPomAIn"), pybind11::arg("xPomBIn"));
		cl.def("setHardDiff", [](Pythia8::Info &o, bool const & a0, bool const & a1, bool const & a2, bool const & a3, double const & a4, double const & a5, double const & a6) -> void { return o.setHardDiff(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("hasUnresBeamsIn"), pybind11::arg("hasPomPsysIn"), pybind11::arg("isHardDiffAIn"), pybind11::arg("isHardDiffBIn"), pybind11::arg("xPomAIn"), pybind11::arg("xPomBIn"), pybind11::arg("tPomAIn"));
		cl.def("setHardDiff", (void (Pythia8::Info::*)(bool, bool, bool, bool, double, double, double, double)) &Pythia8::Info::setHardDiff, "C++: Pythia8::Info::setHardDiff(bool, bool, bool, bool, double, double, double, double) --> void", pybind11::arg("hasUnresBeamsIn"), pybind11::arg("hasPomPsysIn"), pybind11::arg("isHardDiffAIn"), pybind11::arg("isHardDiffBIn"), pybind11::arg("xPomAIn"), pybind11::arg("xPomBIn"), pybind11::arg("tPomAIn"), pybind11::arg("tPomBIn"));
		cl.def("reassignDiffSystem", (void (Pythia8::Info::*)(int, int)) &Pythia8::Info::reassignDiffSystem, "C++: Pythia8::Info::reassignDiffSystem(int, int) --> void", pybind11::arg("iDSold"), pybind11::arg("iDSnew"));
		cl.def("setHasUnresolvedBeams", (void (Pythia8::Info::*)(bool)) &Pythia8::Info::setHasUnresolvedBeams, "C++: Pythia8::Info::setHasUnresolvedBeams(bool) --> void", pybind11::arg("hasUnresBeamsIn"));
		cl.def("setHasPomPsystem", (void (Pythia8::Info::*)(bool)) &Pythia8::Info::setHasPomPsystem, "C++: Pythia8::Info::setHasPomPsystem(bool) --> void", pybind11::arg("hasPomPsysIn"));
		cl.def("numberOfWeights", (int (Pythia8::Info::*)() const) &Pythia8::Info::numberOfWeights, "C++: Pythia8::Info::numberOfWeights() const --> int");
		cl.def("weightValueByIndex", [](Pythia8::Info const &o) -> double { return o.weightValueByIndex(); }, "");
		cl.def("weightValueByIndex", (double (Pythia8::Info::*)(int) const) &Pythia8::Info::weightValueByIndex, "C++: Pythia8::Info::weightValueByIndex(int) const --> double", pybind11::arg("key"));
		cl.def("weightNameByIndex", [](Pythia8::Info const &o) -> std::string { return o.weightNameByIndex(); }, "");
		cl.def("weightNameByIndex", (std::string (Pythia8::Info::*)(int) const) &Pythia8::Info::weightNameByIndex, "C++: Pythia8::Info::weightNameByIndex(int) const --> std::string", pybind11::arg("key"));
		cl.def("weightValueVector", (class std::vector<double, class std::allocator<double> > (Pythia8::Info::*)() const) &Pythia8::Info::weightValueVector, "C++: Pythia8::Info::weightValueVector() const --> class std::vector<double, class std::allocator<double> >");
		cl.def("weightNameVector", (class std::vector<std::string, class std::allocator<std::string > > (Pythia8::Info::*)() const) &Pythia8::Info::weightNameVector, "C++: Pythia8::Info::weightNameVector() const --> class std::vector<std::string, class std::allocator<std::string > >");
	}
}
