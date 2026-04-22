void selection_30()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo61","canvas_plotflow_tempo61",0,0,700,500);
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
  TH1F* S31_M_0 = new TH1F("S31_M_0","S31_M_0",40,0.0,500.0);
  // Content
  S31_M_0->SetBinContent(0,0.0); // underflow
  S31_M_0->SetBinContent(1,0.0);
  S31_M_0->SetBinContent(2,0.0);
  S31_M_0->SetBinContent(3,0.0);
  S31_M_0->SetBinContent(4,0.0);
  S31_M_0->SetBinContent(5,300.2165855400007);
  S31_M_0->SetBinContent(6,1801.2999132399848);
  S31_M_0->SetBinContent(7,9006.49756620002);
  S31_M_0->SetBinContent(8,20414.729016719986);
  S31_M_0->SetBinContent(9,23717.10885766012);
  S31_M_0->SetBinContent(10,41129.678018979816);
  S31_M_0->SetBinContent(11,54339.20738273986);
  S31_M_0->SetBinContent(12,65147.00686217977);
  S31_M_0->SetBinContent(13,75354.36637054);
  S31_M_0->SetBinContent(14,76855.44629824015);
  S31_M_0->SetBinContent(15,91566.06558969988);
  S31_M_0->SetBinContent(16,82259.34603796012);
  S31_M_0->SetBinContent(17,92466.71554631986);
  S31_M_0->SetBinContent(18,81959.13605241978);
  S31_M_0->SetBinContent(19,102674.09505467914);
  S31_M_0->SetBinContent(20,87363.02579214022);
  S31_M_0->SetBinContent(21,81058.4860957998);
  S31_M_0->SetBinContent(22,84060.64595120009);
  S31_M_0->SetBinContent(23,78656.74621148013);
  S31_M_0->SetBinContent(24,85861.94586444007);
  S31_M_0->SetBinContent(25,73252.84647176019);
  S31_M_0->SetBinContent(26,74153.49642838018);
  S31_M_0->SetBinContent(27,69950.46663082005);
  S31_M_0->SetBinContent(28,64846.78687663993);
  S31_M_0->SetBinContent(29,59743.10712245981);
  S31_M_0->SetBinContent(30,59442.88713691998);
  S31_M_0->SetBinContent(31,50436.38757072006);
  S31_M_0->SetBinContent(32,57341.36723814016);
  S31_M_0->SetBinContent(33,52237.68748396004);
  S31_M_0->SetBinContent(34,52537.90746949988);
  S31_M_0->SetBinContent(35,48635.087657480064);
  S31_M_0->SetBinContent(36,42330.53796114012);
  S31_M_0->SetBinContent(37,47434.21771532024);
  S31_M_0->SetBinContent(38,44732.27784545978);
  S31_M_0->SetBinContent(39,35125.33830818018);
  S31_M_0->SetBinContent(40,33924.478366019874);
  S31_M_0->SetBinContent(41,900049.3566489205); // overflow
  S31_M_0->SetEntries(10000);
  // Style
  S31_M_0->SetLineColor(9);
  S31_M_0->SetLineStyle(1);
  S31_M_0->SetLineWidth(1);
  S31_M_0->SetFillColor(9);
  S31_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_62","mystack");
  stack->Add(S31_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l-_{1} p_{1} p_{2} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_30.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_30.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_30.eps");

}
