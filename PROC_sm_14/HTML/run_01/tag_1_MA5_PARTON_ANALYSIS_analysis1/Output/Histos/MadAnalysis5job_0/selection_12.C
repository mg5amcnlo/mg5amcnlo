void selection_12()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo25","canvas_plotflow_tempo25",0,0,700,500);
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
  TH1F* S13_ETA_0 = new TH1F("S13_ETA_0","S13_ETA_0",40,-10.0,10.0);
  // Content
  S13_ETA_0->SetBinContent(0,0.0); // underflow
  S13_ETA_0->SetBinContent(1,0.0);
  S13_ETA_0->SetBinContent(2,0.0);
  S13_ETA_0->SetBinContent(3,0.0);
  S13_ETA_0->SetBinContent(4,0.0);
  S13_ETA_0->SetBinContent(5,0.0);
  S13_ETA_0->SetBinContent(6,0.0);
  S13_ETA_0->SetBinContent(7,0.0);
  S13_ETA_0->SetBinContent(8,0.0);
  S13_ETA_0->SetBinContent(9,0.0);
  S13_ETA_0->SetBinContent(10,0.0);
  S13_ETA_0->SetBinContent(11,26719.279732999974);
  S13_ETA_0->SetBinContent(12,46833.789532);
  S13_ETA_0->SetBinContent(13,76555.23923499994);
  S13_ETA_0->SetBinContent(14,95769.09904299997);
  S13_ETA_0->SetBinContent(15,136298.29863800036);
  S13_ETA_0->SetBinContent(16,170522.9982960003);
  S13_ETA_0->SetBinContent(17,214955.09785199986);
  S13_ETA_0->SetBinContent(18,232367.6976779995);
  S13_ETA_0->SetBinContent(19,262389.2973780001);
  S13_ETA_0->SetBinContent(20,268093.3973210003);
  S13_ETA_0->SetBinContent(21,259086.89741100027);
  S13_ETA_0->SetBinContent(22,259987.59740199978);
  S13_ETA_0->SetBinContent(23,215855.69784300038);
  S13_ETA_0->SetBinContent(24,188836.29811299942);
  S13_ETA_0->SetBinContent(25,173224.9982689998);
  S13_ETA_0->SetBinContent(26,125190.29874900023);
  S13_ETA_0->SetBinContent(27,102674.09897399979);
  S13_ETA_0->SetBinContent(28,69950.46930099999);
  S13_ETA_0->SetBinContent(29,49235.51950800003);
  S13_ETA_0->SetBinContent(30,27619.929723999972);
  S13_ETA_0->SetBinContent(31,0.0);
  S13_ETA_0->SetBinContent(32,0.0);
  S13_ETA_0->SetBinContent(33,0.0);
  S13_ETA_0->SetBinContent(34,0.0);
  S13_ETA_0->SetBinContent(35,0.0);
  S13_ETA_0->SetBinContent(36,0.0);
  S13_ETA_0->SetBinContent(37,0.0);
  S13_ETA_0->SetBinContent(38,0.0);
  S13_ETA_0->SetBinContent(39,0.0);
  S13_ETA_0->SetBinContent(40,0.0);
  S13_ETA_0->SetBinContent(41,0.0); // overflow
  S13_ETA_0->SetEntries(10000);
  // Style
  S13_ETA_0->SetLineColor(9);
  S13_ETA_0->SetLineStyle(1);
  S13_ETA_0->SetLineWidth(1);
  S13_ETA_0->SetFillColor(9);
  S13_ETA_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_26","mystack");
  stack->Add(S13_ETA_0);
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
  stack->GetXaxis()->SetTitle("#eta [ p_{3} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_12.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_12.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_12.eps");

}
