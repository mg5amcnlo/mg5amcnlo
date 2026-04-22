void selection_16()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo33","canvas_plotflow_tempo33",0,0,700,500);
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
  TH1F* S17_M_0 = new TH1F("S17_M_0","S17_M_0",40,0.0,500.0);
  // Content
  S17_M_0->SetBinContent(0,0.0); // underflow
  S17_M_0->SetBinContent(1,0.0);
  S17_M_0->SetBinContent(2,0.0);
  S17_M_0->SetBinContent(3,1501.0829375000028);
  S17_M_0->SetBinContent(4,7805.631674999997);
  S17_M_0->SetBinContent(5,25218.198949999813);
  S17_M_0->SetBinContent(6,65747.43726249992);
  S17_M_0->SetBinContent(7,89464.54627500003);
  S17_M_0->SetBinContent(8,113481.89527499914);
  S17_M_0->SetBinContent(9,123088.79487500047);
  S17_M_0->SetBinContent(10,138700.09422499896);
  S17_M_0->SetBinContent(11,142302.69407499896);
  S17_M_0->SetBinContent(12,136898.79429999896);
  S17_M_0->SetBinContent(13,126691.39472500043);
  S17_M_0->SetBinContent(14,122788.59488749978);
  S17_M_0->SetBinContent(15,138700.09422499896);
  S17_M_0->SetBinContent(16,121887.89492500186);
  S17_M_0->SetBinContent(17,112881.39530000192);
  S17_M_0->SetBinContent(18,93367.36611249985);
  S17_M_0->SetBinContent(19,101473.19577500063);
  S17_M_0->SetBinContent(20,86762.59638750005);
  S17_M_0->SetBinContent(21,86462.3764000002);
  S17_M_0->SetBinContent(22,84060.64650000006);
  S17_M_0->SetBinContent(23,61244.18744999996);
  S17_M_0->SetBinContent(24,66648.08722499991);
  S17_M_0->SetBinContent(25,64246.3573249998);
  S17_M_0->SetBinContent(26,53138.337787500015);
  S17_M_0->SetBinContent(27,44432.058149999946);
  S17_M_0->SetBinContent(28,51036.817875000175);
  S17_M_0->SetBinContent(29,43831.62817499981);
  S17_M_0->SetBinContent(30,42630.75822499996);
  S17_M_0->SetBinContent(31,34524.90856250002);
  S17_M_0->SetBinContent(32,32723.608637500034);
  S17_M_0->SetBinContent(33,32723.608637500034);
  S17_M_0->SetBinContent(34,31522.738687500183);
  S17_M_0->SetBinContent(35,29721.43876250019);
  S17_M_0->SetBinContent(36,25218.198949999813);
  S17_M_0->SetBinContent(37,21015.15912500012);
  S17_M_0->SetBinContent(38,24617.75897500009);
  S17_M_0->SetBinContent(39,21015.15912500012);
  S17_M_0->SetBinContent(40,21315.37911249998);
  S17_M_0->SetBinContent(41,381275.0841249999); // overflow
  S17_M_0->SetEntries(10000);
  // Style
  S17_M_0->SetLineColor(9);
  S17_M_0->SetLineStyle(1);
  S17_M_0->SetLineWidth(1);
  S17_M_0->SetFillColor(9);
  S17_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_34","mystack");
  stack->Add(S17_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} p_{1} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_16.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_16.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_16.eps");

}
