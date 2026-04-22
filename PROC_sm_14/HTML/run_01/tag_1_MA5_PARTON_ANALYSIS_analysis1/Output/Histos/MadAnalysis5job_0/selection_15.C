void selection_15()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo31","canvas_plotflow_tempo31",0,0,700,500);
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
  TH1F* S16_M_0 = new TH1F("S16_M_0","S16_M_0",40,0.0,500.0);
  // Content
  S16_M_0->SetBinContent(0,0.0); // underflow
  S16_M_0->SetBinContent(1,0.0);
  S16_M_0->SetBinContent(2,0.0);
  S16_M_0->SetBinContent(3,0.0);
  S16_M_0->SetBinContent(4,0.0);
  S16_M_0->SetBinContent(5,600.4331903200003);
  S16_M_0->SetBinContent(6,600.4331903200003);
  S16_M_0->SetBinContent(7,6904.981888679999);
  S16_M_0->SetBinContent(8,15611.259748320057);
  S16_M_0->SetBinContent(9,29421.22952567996);
  S16_M_0->SetBinContent(10,39928.80935627998);
  S16_M_0->SetBinContent(11,63946.138969079955);
  S16_M_0->SetBinContent(12,67548.73891099995);
  S16_M_0->SetBinContent(13,73553.06881419999);
  S16_M_0->SetBinContent(14,73553.06881419999);
  S16_M_0->SetBinContent(15,85861.94861575999);
  S16_M_0->SetBinContent(16,89464.54855767998);
  S16_M_0->SetBinContent(17,92766.92850444002);
  S16_M_0->SetBinContent(18,102073.59835440075);
  S16_M_0->SetBinContent(19,93067.14849959996);
  S16_M_0->SetBinContent(20,85561.72862060004);
  S16_M_0->SetBinContent(21,78056.31874159997);
  S16_M_0->SetBinContent(22,77155.66875611997);
  S16_M_0->SetBinContent(23,80157.8387077199);
  S16_M_0->SetBinContent(24,84961.29863028);
  S16_M_0->SetBinContent(25,82859.78866415989);
  S16_M_0->SetBinContent(26,70550.89886260004);
  S16_M_0->SetBinContent(27,70851.11885775998);
  S16_M_0->SetBinContent(28,61544.40900779991);
  S16_M_0->SetBinContent(29,69350.03888195993);
  S16_M_0->SetBinContent(30,68749.59889164005);
  S16_M_0->SetBinContent(31,63946.138969079955);
  S16_M_0->SetBinContent(32,50436.38918688);
  S16_M_0->SetBinContent(33,48635.08921592);
  S16_M_0->SetBinContent(34,46533.57924979991);
  S16_M_0->SetBinContent(35,43531.40929819997);
  S16_M_0->SetBinContent(36,47734.43923044001);
  S16_M_0->SetBinContent(37,46833.78924496001);
  S16_M_0->SetBinContent(38,36626.42940951994);
  S16_M_0->SetBinContent(39,42930.97930787992);
  S16_M_0->SetBinContent(40,38427.729380479934);
  S16_M_0->SetBinContent(41,871828.9859446405); // overflow
  S16_M_0->SetEntries(10000);
  // Style
  S16_M_0->SetLineColor(9);
  S16_M_0->SetLineStyle(1);
  S16_M_0->SetLineWidth(1);
  S16_M_0->SetFillColor(9);
  S16_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_32","mystack");
  stack->Add(S16_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} p_{1} p_{2} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_15.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_15.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_15.eps");

}
