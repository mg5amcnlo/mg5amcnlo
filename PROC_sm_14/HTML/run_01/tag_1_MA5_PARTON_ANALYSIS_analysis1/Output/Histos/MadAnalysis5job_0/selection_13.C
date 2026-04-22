void selection_13()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo27","canvas_plotflow_tempo27",0,0,700,500);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  canvas->SetHighLightColor(2);
  canvas->SetFillColor(0);
  canvas->SetBorderMode(0);
  canvas->SetBorderSize(3);
  canvas->SetFrameBorderMode(0);
  canvas->SetFrameBorderSize(0);
  canvas->SetTickx(1);
  canvas->SetTicky(1);
  canvas->SetLeftMargin(0.14);
  canvas->SetRightMargin(0.05);
  canvas->SetBottomMargin(0.15);
  canvas->SetTopMargin(0.05);

  // Creating a new TH1F
  TH1F* S14_M_0 = new TH1F("S14_M_0","S14_M_0",40,0.0,500.0);
  // Content
  S14_M_0->SetBinContent(0,0.0); // underflow
  S14_M_0->SetBinContent(1,19514.079135499993);
  S14_M_0->SetBinContent(2,57641.58744639998);
  S14_M_0->SetBinContent(3,99371.69559769995);
  S14_M_0->SetBinContent(4,157313.49303080022);
  S14_M_0->SetBinContent(5,195741.1913284014);
  S14_M_0->SetBinContent(6,208950.7907431983);
  S14_M_0->SetBinContent(7,205648.39088949907);
  S14_M_0->SetBinContent(8,201145.09108900136);
  S14_M_0->SetBinContent(9,193639.69142150067);
  S14_M_0->SetBinContent(10,163317.79276480165);
  S14_M_0->SetBinContent(11,151309.1932967988);
  S14_M_0->SetBinContent(12,139000.2938420996);
  S14_M_0->SetBinContent(13,124289.694493799);
  S14_M_0->SetBinContent(14,115583.39487949981);
  S14_M_0->SetBinContent(15,94568.22581050012);
  S14_M_0->SetBinContent(16,83460.21630259992);
  S14_M_0->SetBinContent(17,77455.88656859982);
  S14_M_0->SetBinContent(18,66648.08704739991);
  S14_M_0->SetBinContent(19,56740.93748629999);
  S14_M_0->SetBinContent(20,51337.03772570004);
  S14_M_0->SetBinContent(21,46833.78792520007);
  S14_M_0->SetBinContent(22,43231.1880848001);
  S14_M_0->SetBinContent(23,36626.428377399854);
  S14_M_0->SetBinContent(24,36626.428377399854);
  S14_M_0->SetBinContent(25,27920.148763099776);
  S14_M_0->SetBinContent(26,28820.788723200214);
  S14_M_0->SetBinContent(27,26419.058829600086);
  S14_M_0->SetBinContent(28,21915.80902910012);
  S14_M_0->SetBinContent(29,20714.949082299834);
  S14_M_0->SetBinContent(30,20414.729095599985);
  S14_M_0->SetBinContent(31,17712.779215300005);
  S14_M_0->SetBinContent(32,14710.609348300179);
  S14_M_0->SetBinContent(33,12308.879454700049);
  S14_M_0->SetBinContent(34,14110.179374900035);
  S14_M_0->SetBinContent(35,9306.7145877);
  S14_M_0->SetBinContent(36,12308.879454700049);
  S14_M_0->SetBinContent(37,6304.548720699994);
  S14_M_0->SetBinContent(38,8105.84864089998);
  S14_M_0->SetBinContent(39,9907.147561100008);
  S14_M_0->SetBinContent(40,9606.931574399983);
  S14_M_0->SetBinContent(41,115583.39487949981); // overflow
  S14_M_0->SetEntries(10000);
  // Style
  S14_M_0->SetLineColor(9);
  S14_M_0->SetLineStyle(1);
  S14_M_0->SetLineWidth(1);
  S14_M_0->SetFillColor(9);
  S14_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_28","mystack");
  stack->Add(S14_M_0);
  stack->Draw("");

  // Y axis
  stack->GetYaxis()->SetLabelSize(0.04);
  stack->GetYaxis()->SetLabelOffset(0.005);
  stack->GetYaxis()->SetTitleSize(0.06);
  stack->GetYaxis()->SetTitleFont(22);
  stack->GetYaxis()->SetTitleOffset(1);
  stack->GetYaxis()->SetTitle("Events  ( L_{int} = 10 fb^{-1} )");

  // X axis
  stack->GetXaxis()->SetLabelSize(0.04);
  stack->GetXaxis()->SetLabelOffset(0.005);
  stack->GetXaxis()->SetTitleSize(0.06);
  stack->GetXaxis()->SetTitleFont(22);
  stack->GetXaxis()->SetTitleOffset(1);
  stack->GetXaxis()->SetTitle("M [ l+_{1} p_{1} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_13.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_13.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_13.eps");

}
