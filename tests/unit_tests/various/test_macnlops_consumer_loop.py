"""Run the actual normal-consumer loop with standard-library-only stubs.

No Pythia/HepMC headers, libraries, native events or showers are involved.
"""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
DRIVERS = ROOT / 'Template/NLO/MCatNLO/srcPythia8'


@unittest.skipUnless(shutil.which('g++'), 'requires a C++ compiler')
class TestMacNLOPSConsumerLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bodies = []
        for name in ('Pythia8.cc', 'Pythia82.cc', 'Pythia83.cc'):
            source = (DRIVERS / name).read_text()
            start = source.index('  double nSelected;')
            bodies.append(source[start:source.index('\n  pythia.stat();', start)])
        if bodies[1:] != bodies[:1] * 2:
            raise AssertionError('Consumer loop must agree across all three Pythia versions')
        cls.temporary = tempfile.TemporaryDirectory(prefix='macnlops-consumer-loop-')
        cls.addClassCleanup(cls.temporary.cleanup)
        work = Path(cls.temporary.name)
        cls.executable = work / 'check'
        stub = r'''
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
using namespace std;
struct State {
  bool continuation=true;
  int requested=3,total=3,fail=0,calls=0,selected=0,read=0;
  double norm=-1.;
  vector<double> source{2.,-3.,5.},written,analysed;
} state;
struct Info {
  int nSelected() const {return state.selected;}
  int nTried() const {return state.calls;}
  double weight() const {return state.source.at(state.calls-1);}
  bool isLHA() const {return false;}
  void list() const {}
};
struct Listing {void list() const {}};
struct Pythia {
  Info info; Listing process,event;
  bool next() {
    ++state.calls;
    if(state.calls==state.fail || state.calls>int(state.source.size()))return false;
    ++state.selected;return true;
  }
  bool flag(const string&) const {return state.continuation;}
  void LHAeventList() const {}
};
namespace HepMC {
struct GenEvent {
  vector<double> values{0.};
  vector<double>& weights(){return values;}
};
struct Pythia8ToHepMC {
  void fill_next_event(Pythia& p,GenEvent* e){e->values[0]=p.info.weight();}
};
struct IO_BaseClass {
  void write_event(GenEvent* e){state.written.push_back(e->weights()[0]);}
};
struct IO_HEPEVT:IO_BaseClass {};
}
struct Reader {
  void lhef_read_wgts_(double* weights){weights[0]=state.source.at(state.read++);}
};
struct {double EVWGT;} cevwgt;
void pyanal_(int&,double* weights){state.analysed.push_back(weights[0]);}
void pyaend_(double& norm){state.norm=norm;}
int consume(bool sum) {
  Pythia pythia;Reader read;
  bool continuation=pythia.flag("Main:spareFlag1");
  int iEventshower=state.requested,iEventtot=state.total;
  int iEventtot_norm=sum?iEventtot:1;
  string evt_norm=sum?"sum":"average";
  int iAbort=0,nAbort=10,iPrintLHA=0,nPrintLHA=0,nstep=5000,cwgtinfo_nn=1;
  double cwgt_ww[1]={0.};
  HepMC::IO_BaseClass* _hepevtio;
  HepMC::Pythia8ToHepMC ToHepMC;
__ACTUAL_LOOP__
  return 0;
}
void array(const vector<double>& values) {
  cout<<'[';
  for(size_t i=0;i<values.size();++i){if(i)cout<<',';cout<<values[i];}
  cout<<']';
}
int main(int argc,char** argv) {
  state.requested=atoi(argv[1]);state.total=state.requested;
  state.fail=atoi(argv[2]);state.continuation=atoi(argv[3]);
  int code=consume(atoi(argv[4]));
  cout<<setprecision(17)<<"{\"code\":"<<code<<",\"calls\":"<<state.calls
      <<",\"read\":"<<state.read<<",\"norm\":"<<state.norm<<",\"written\":";
  array(state.written);cout<<",\"analysed\":";array(state.analysed);cout<<"}\n";
}
'''
        source = work / 'check.cc'
        source.write_text(stub.replace('__ACTUAL_LOOP__', bodies[0]))
        result = subprocess.run(['nice', '-n', '19', 'g++', '-std=c++11', '-O0',
            str(source), '-o', str(cls.executable)], cwd=work, text=True,
            capture_output=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check(self, count, fail=0, continuation=True, sum_norm=True):
        result = subprocess.run([str(self.executable), str(count), str(fail),
            str(int(continuation)), str(int(sum_norm))], text=True,
            capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_one_event_is_written_and_norm_is_nonzero(self):
        self.assertEqual(self.check(1), dict(code=0, calls=1, read=1,
            norm=1., written=[2.], analysed=[2.]))

    def test_three_signed_events_are_written_without_reading_a_fourth(self):
        self.assertEqual(self.check(3), dict(code=0, calls=3, read=3,
            norm=3., written=[2., -3., 5.], analysed=[2., -3., 5.]))

    def test_average_normalization_uses_processed_events(self):
        report = self.check(3, sum_norm=False)
        self.assertEqual(report['norm'], 1.)
        self.assertEqual(report['written'], [2., -3., 5.])

    def test_failed_next_stops_before_consuming_any_replacement(self):
        for failed in (1, 2):
            report = self.check(3, fail=failed)
            self.assertEqual(report['code'], 1)
            self.assertEqual(report['calls'], failed)
            self.assertEqual(report['written'], [2.][:failed - 1])
            self.assertEqual(report['analysed'], report['written'])
            self.assertEqual(report['read'], failed - 1)

    def test_opt_out_preserves_legacy_stop_and_normalization(self):
        self.assertEqual(self.check(3, continuation=False), dict(code=0,
            calls=3, read=2, norm=2., written=[2., -3.], analysed=[2., -3.]))


if __name__ == '__main__':
    unittest.main()
