void selection_27()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo55","canvas_plotflow_tempo55",0,0,700,500);
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
  TH1F* S28_M_0 = new TH1F("S28_M_0","S28_M_0",40,0.0,500.0);
  // Content
  S28_M_0->SetBinContent(0,0.0); // underflow
  S28_M_0->SetBinContent(1,600.4331865600004);
  S28_M_0->SetBinContent(2,18913.649576639917);
  S28_M_0->SetBinContent(3,29421.229341439943);
  S28_M_0->SetBinContent(4,43231.189032320035);
  S28_M_0->SetBinContent(5,59142.668676160036);
  S28_M_0->SetBinContent(6,60043.31865600003);
  S28_M_0->SetBinContent(7,49835.95888447993);
  S28_M_0->SetBinContent(8,63946.13856863994);
  S28_M_0->SetBinContent(9,263590.1940998396);
  S28_M_0->SetBinContent(10,386078.591358079);
  S28_M_0->SetBinContent(11,320030.8928364801);
  S28_M_0->SetBinContent(12,248279.09444256077);
  S28_M_0->SetBinContent(13,189436.69575967954);
  S28_M_0->SetBinContent(14,146205.4967273597);
  S28_M_0->SetBinContent(15,125490.49719104094);
  S28_M_0->SetBinContent(16,102073.59771520105);
  S28_M_0->SetBinContent(17,91265.84795711997);
  S28_M_0->SetBinContent(18,84661.08810495984);
  S28_M_0->SetBinContent(19,61844.618615680025);
  S28_M_0->SetBinContent(20,58842.458682879886);
  S28_M_0->SetBinContent(21,47134.00894495994);
  S28_M_0->SetBinContent(22,45332.70898527995);
  S28_M_0->SetBinContent(23,42930.979039039885);
  S28_M_0->SetBinContent(24,32423.38927424008);
  S28_M_0->SetBinContent(25,25818.62942207996);
  S28_M_0->SetBinContent(26,27319.709388480027);
  S28_M_0->SetBinContent(27,24317.54945567989);
  S28_M_0->SetBinContent(28,27619.929381759954);
  S28_M_0->SetBinContent(29,18012.99959679992);
  S28_M_0->SetBinContent(30,17712.779603519994);
  S28_M_0->SetBinContent(31,19514.07956319999);
  S28_M_0->SetBinContent(32,18613.429583359994);
  S28_M_0->SetBinContent(33,13809.959690880087);
  S28_M_0->SetBinContent(34,12909.309711040092);
  S28_M_0->SetBinContent(35,10507.579764800028);
  S28_M_0->SetBinContent(36,9606.931784959988);
  S28_M_0->SetBinContent(37,10507.579764800028);
  S28_M_0->SetBinContent(38,12609.099717759944);
  S28_M_0->SetBinContent(39,11108.0097513601);
  S28_M_0->SetBinContent(40,8706.28180511999);
  S28_M_0->SetBinContent(41,162717.39635776); // overflow
  S28_M_0->SetEntries(10000);
  // Style
  S28_M_0->SetLineColor(9);
  S28_M_0->SetLineStyle(1);
  S28_M_0->SetLineWidth(1);
  S28_M_0->SetFillColor(9);
  S28_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_56","mystack");
  stack->Add(S28_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} l-_{1} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_27.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_27.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_27.eps");

}
