void selection_10()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo21","canvas_plotflow_tempo21",0,0,700,500);
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
  TH1F* S11_ETA_0 = new TH1F("S11_ETA_0","S11_ETA_0",40,-10.0,10.0);
  // Content
  S11_ETA_0->SetBinContent(0,0.0); // underflow
  S11_ETA_0->SetBinContent(1,0.0);
  S11_ETA_0->SetBinContent(2,0.0);
  S11_ETA_0->SetBinContent(3,0.0);
  S11_ETA_0->SetBinContent(4,0.0);
  S11_ETA_0->SetBinContent(5,0.0);
  S11_ETA_0->SetBinContent(6,0.0);
  S11_ETA_0->SetBinContent(7,0.0);
  S11_ETA_0->SetBinContent(8,0.0);
  S11_ETA_0->SetBinContent(9,0.0);
  S11_ETA_0->SetBinContent(10,0.0);
  S11_ETA_0->SetBinContent(11,13209.529120000087);
  S11_ETA_0->SetBinContent(12,32123.17785999989);
  S11_ETA_0->SetBinContent(13,51937.46654000036);
  S11_ETA_0->SetBinContent(14,89464.5440400002);
  S11_ETA_0->SetBinContent(15,107177.29286000223);
  S11_ETA_0->SetBinContent(16,166019.7889399994);
  S11_ETA_0->SetBinContent(17,228164.5848000021);
  S11_ETA_0->SetBinContent(18,243475.68377999862);
  S11_ETA_0->SetBinContent(19,291210.08060000144);
  S11_ETA_0->SetBinContent(20,290309.48065999814);
  S11_ETA_0->SetBinContent(21,303819.1797600013);
  S11_ETA_0->SetBinContent(22,275598.8816399972);
  S11_ETA_0->SetBinContent(23,247078.28353999855);
  S11_ETA_0->SetBinContent(24,204147.2864000001);
  S11_ETA_0->SetBinContent(25,157613.68950000172);
  S11_ETA_0->SetBinContent(26,109879.29267999888);
  S11_ETA_0->SetBinContent(27,90665.41395999995);
  S11_ETA_0->SetBinContent(28,51637.256559999914);
  S11_ETA_0->SetBinContent(29,31522.737900000342);
  S11_ETA_0->SetBinContent(30,17112.348859999824);
  S11_ETA_0->SetBinContent(31,0.0);
  S11_ETA_0->SetBinContent(32,0.0);
  S11_ETA_0->SetBinContent(33,0.0);
  S11_ETA_0->SetBinContent(34,0.0);
  S11_ETA_0->SetBinContent(35,0.0);
  S11_ETA_0->SetBinContent(36,0.0);
  S11_ETA_0->SetBinContent(37,0.0);
  S11_ETA_0->SetBinContent(38,0.0);
  S11_ETA_0->SetBinContent(39,0.0);
  S11_ETA_0->SetBinContent(40,0.0);
  S11_ETA_0->SetBinContent(41,0.0); // overflow
  S11_ETA_0->SetEntries(10000);
  // Style
  S11_ETA_0->SetLineColor(9);
  S11_ETA_0->SetLineStyle(1);
  S11_ETA_0->SetLineWidth(1);
  S11_ETA_0->SetFillColor(9);
  S11_ETA_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_22","mystack");
  stack->Add(S11_ETA_0);
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
  stack->GetXaxis()->SetTitle("#eta [ p_{2} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_10.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_10.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_10.eps");

}
