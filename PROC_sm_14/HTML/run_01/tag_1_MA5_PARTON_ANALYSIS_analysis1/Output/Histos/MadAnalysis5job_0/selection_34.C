void selection_34()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo69","canvas_plotflow_tempo69",0,0,700,500);
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
  TH1F* S35_M_0 = new TH1F("S35_M_0","S35_M_0",40,0.0,500.0);
  // Content
  S35_M_0->SetBinContent(0,0.0); // underflow
  S35_M_0->SetBinContent(1,51036.82005099998);
  S35_M_0->SetBinContent(2,166320.00016619996);
  S35_M_0->SetBinContent(3,261788.90026159995);
  S35_M_0->SetBinContent(4,330538.5003302999);
  S35_M_0->SetBinContent(5,332039.60033179994);
  S35_M_0->SetBinContent(6,299015.7002987999);
  S35_M_0->SetBinContent(7,232968.10023279994);
  S35_M_0->SetBinContent(8,190337.3001901999);
  S35_M_0->SetBinContent(9,159415.00015929993);
  S35_M_0->SetBinContent(10,120086.60011999993);
  S35_M_0->SetBinContent(11,111080.10011099993);
  S35_M_0->SetBinContent(12,90665.42009059998);
  S35_M_0->SetBinContent(13,71751.77007169998);
  S35_M_0->SetBinContent(14,68449.39006839998);
  S35_M_0->SetBinContent(15,48334.870048299985);
  S35_M_0->SetBinContent(16,48334.870048299985);
  S35_M_0->SetBinContent(17,39028.160038999995);
  S35_M_0->SetBinContent(18,33624.260033599996);
  S35_M_0->SetBinContent(19,25818.630025799994);
  S35_M_0->SetBinContent(20,31222.530031199993);
  S35_M_0->SetBinContent(21,26118.850026099997);
  S35_M_0->SetBinContent(22,21915.810021899993);
  S35_M_0->SetBinContent(23,22516.250022499997);
  S35_M_0->SetBinContent(24,18613.430018599996);
  S35_M_0->SetBinContent(25,11708.450011699999);
  S35_M_0->SetBinContent(26,13509.7500135);
  S35_M_0->SetBinContent(27,11408.230011399995);
  S35_M_0->SetBinContent(28,13509.7500135);
  S35_M_0->SetBinContent(29,9006.498008999997);
  S35_M_0->SetBinContent(30,10507.580010499996);
  S35_M_0->SetBinContent(31,9907.148009899996);
  S35_M_0->SetBinContent(32,5403.899005399999);
  S35_M_0->SetBinContent(33,9907.148009899996);
  S35_M_0->SetBinContent(34,5704.116005699999);
  S35_M_0->SetBinContent(35,7205.199007199998);
  S35_M_0->SetBinContent(36,8406.065008399999);
  S35_M_0->SetBinContent(37,4503.249004499999);
  S35_M_0->SetBinContent(38,6304.549006299999);
  S35_M_0->SetBinContent(39,3302.383003299999);
  S35_M_0->SetBinContent(40,3302.383003299999);
  S35_M_0->SetBinContent(41,67548.7400675); // overflow
  S35_M_0->SetEntries(10000);
  // Style
  S35_M_0->SetLineColor(9);
  S35_M_0->SetLineStyle(1);
  S35_M_0->SetLineWidth(1);
  S35_M_0->SetFillColor(9);
  S35_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_70","mystack");
  stack->Add(S35_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l-_{1} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_34.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_34.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_34.eps");

}
