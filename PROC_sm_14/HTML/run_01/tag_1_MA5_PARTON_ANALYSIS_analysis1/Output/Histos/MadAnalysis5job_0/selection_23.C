void selection_23()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo47","canvas_plotflow_tempo47",0,0,700,500);
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
  TH1F* S24_M_0 = new TH1F("S24_M_0","S24_M_0",40,0.0,500.0);
  // Content
  S24_M_0->SetBinContent(0,0.0); // underflow
  S24_M_0->SetBinContent(1,0.0);
  S24_M_0->SetBinContent(2,0.0);
  S24_M_0->SetBinContent(3,0.0);
  S24_M_0->SetBinContent(4,0.0);
  S24_M_0->SetBinContent(5,0.0);
  S24_M_0->SetBinContent(6,0.0);
  S24_M_0->SetBinContent(7,300.21655376000706);
  S24_M_0->SetBinContent(8,900.6496612800212);
  S24_M_0->SetBinContent(9,4503.248306400105);
  S24_M_0->SetBinContent(10,5103.68121392015);
  S24_M_0->SetBinContent(11,8406.063705280167);
  S24_M_0->SetBinContent(12,11408.22824288039);
  S24_M_0->SetBinContent(13,14110.177826720363);
  S24_M_0->SetBinContent(14,27019.49583839971);
  S24_M_0->SetBinContent(15,37527.07422000011);
  S24_M_0->SetBinContent(16,42030.32352640007);
  S24_M_0->SetBinContent(17,51637.252046720474);
  S24_M_0->SetBinContent(18,61544.40052080037);
  S24_M_0->SetBinContent(19,67848.93954976184);
  S24_M_0->SetBinContent(20,73252.83871744179);
  S24_M_0->SetBinContent(21,76855.43816256174);
  S24_M_0->SetBinContent(22,90665.40603552108);
  S24_M_0->SetBinContent(23,77155.65811632122);
  S24_M_0->SetBinContent(24,78356.5279313607);
  S24_M_0->SetBinContent(25,69650.23927232182);
  S24_M_0->SetBinContent(26,80458.0376076817);
  S24_M_0->SetBinContent(27,78656.73788512172);
  S24_M_0->SetBinContent(28,72051.97890240078);
  S24_M_0->SetBinContent(29,80458.0376076817);
  S24_M_0->SetBinContent(30,72051.97890240078);
  S24_M_0->SetBinContent(31,58542.23098320091);
  S24_M_0->SetBinContent(32,66648.07973472083);
  S24_M_0->SetBinContent(33,63946.130150880854);
  S24_M_0->SetBinContent(34,57641.58112192092);
  S24_M_0->SetBinContent(35,66347.85978096134);
  S24_M_0->SetBinContent(36,51036.81213920151);
  S24_M_0->SetBinContent(37,52237.68195424099);
  S24_M_0->SetBinContent(38,56140.50135312043);
  S24_M_0->SetBinContent(39,50436.382231681004);
  S24_M_0->SetBinContent(40,48935.3024628805);
  S24_M_0->SetBinContent(41,1248300.8077340513); // overflow
  S24_M_0->SetEntries(10000);
  // Style
  S24_M_0->SetLineColor(9);
  S24_M_0->SetLineStyle(1);
  S24_M_0->SetLineWidth(1);
  S24_M_0->SetFillColor(9);
  S24_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_48","mystack");
  stack->Add(S24_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} l-_{1} p_{1} p_{2} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_23.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_23.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_23.eps");

}
