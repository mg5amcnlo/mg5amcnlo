#ifndef analysis_user_h
#define analysis_user_h

#include "SampleAnalyzer/Process/Analyzer/AnalyzerBase.h"

namespace MA5
{
class user : public AnalyzerBase
{
    INIT_ANALYSIS(user, "MadAnalysis5job")

    public : 
        MAbool Initialize(const MA5::Configuration& cfg,
                          const std::map<std::string,std::string>& parameters);
        void Finalize(const SampleFormat& summary, const std::vector<SampleFormat>& files);
        MAbool Execute(SampleFormat& sample, const EventFormat& event);

    private : 
  // Declaring particle containers
  std::vector<const MCParticleFormat*> _P_l_p_I1I_PTorderingfinalstate_REG_;
  std::vector<const MCParticleFormat*> _P_l_m_I1I_PTorderingfinalstate_REG_;
  std::vector<const MCParticleFormat*> _P_p_I1I_PTorderingfinalstate_REG_;
  std::vector<const MCParticleFormat*> _P_p_I2I_PTorderingfinalstate_REG_;
  std::vector<const MCParticleFormat*> _P_p_I3I_PTorderingfinalstate_REG_;
  std::vector<const MCParticleFormat*> _P_l_pPTorderingfinalstate_REG_;
  MAbool isP__l_pPTorderingfinalstate(const MCParticleFormat* part) const {
     if ( part==0 ) return false;
     if ( !PHYSICS->Id->IsFinalState(part) ) return false;
     if ( (part->pdgid()!=-13)&&(part->pdgid()!=-11) ) return false;
     return true; }
  std::vector<const MCParticleFormat*> _P_l_mPTorderingfinalstate_REG_;
  MAbool isP__l_mPTorderingfinalstate(const MCParticleFormat* part) const {
     if ( part==0 ) return false;
     if ( !PHYSICS->Id->IsFinalState(part) ) return false;
     if ( (part->pdgid()!=11)&&(part->pdgid()!=13) ) return false;
     return true; }
  std::vector<const MCParticleFormat*> _P_pPTorderingfinalstate_REG_;
  MAbool isP__pPTorderingfinalstate(const MCParticleFormat* part) const {
     if ( part==0 ) return false;
     if ( !PHYSICS->Id->IsFinalState(part) ) return false;
     if ( (part->pdgid()!=-4)&&(part->pdgid()!=-3)&&(part->pdgid()!=-2)&&(part->pdgid()!=-1)&&(part->pdgid()!=1)&&(part->pdgid()!=2)&&(part->pdgid()!=3)&&(part->pdgid()!=4)&&(part->pdgid()!=21) ) return false;
     return true; }
};
}

#endif