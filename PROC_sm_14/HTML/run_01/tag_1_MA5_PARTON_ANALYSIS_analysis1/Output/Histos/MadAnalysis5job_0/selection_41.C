void selection_41()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo83","canvas_plotflow_tempo83",0,0,700,500);
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
  TH1F* S42_DELTAR_0 = new TH1F("S42_DELTAR_0","S42_DELTAR_0",40,0.0,10.0);
  // Content
  S42_DELTAR_0->SetBinContent(0,0.0); // underflow
  S42_DELTAR_0->SetBinContent(1,18013.001029600295);
  S42_DELTAR_0->SetBinContent(2,47734.44272844021);
  S42_DELTAR_0->SetBinContent(3,68749.60392964017);
  S42_DELTAR_0->SetBinContent(4,100272.30573143784);
  S42_DELTAR_0->SetBinContent(5,128792.90736163924);
  S42_DELTAR_0->SetBinContent(6,147706.6084427224);
  S42_DELTAR_0->SetBinContent(7,170823.30976404375);
  S42_DELTAR_0->SetBinContent(8,198743.41135992133);
  S42_DELTAR_0->SetBinContent(9,221259.61264691886);
  S42_DELTAR_0->SetBinContent(10,223060.9127498789);
  S42_DELTAR_0->SetBinContent(11,255184.11458600036);
  S42_DELTAR_0->SetBinContent(12,263590.21506648243);
  S42_DELTAR_0->SetBinContent(13,259086.91480907946);
  S42_DELTAR_0->SetBinContent(14,175926.91005575907);
  S42_DELTAR_0->SetBinContent(15,148907.40851135863);
  S42_DELTAR_0->SetBinContent(16,121887.90696695818);
  S42_DELTAR_0->SetBinContent(17,101773.4058172388);
  S42_DELTAR_0->SetBinContent(18,75654.59432432066);
  S42_DELTAR_0->SetBinContent(19,61844.62353496025);
  S42_DELTAR_0->SetBinContent(20,55239.86315744052);
  S42_DELTAR_0->SetBinContent(21,47434.22271128002);
  S42_DELTAR_0->SetBinContent(22,32423.391853279958);
  S42_DELTAR_0->SetBinContent(23,26419.061510080053);
  S42_DELTAR_0->SetBinContent(24,16812.130960960087);
  S42_DELTAR_0->SetBinContent(25,14710.61084083986);
  S42_DELTAR_0->SetBinContent(26,10207.360583439786);
  S42_DELTAR_0->SetBinContent(27,4803.466274560041);
  S42_DELTAR_0->SetBinContent(28,2101.5161201199962);
  S42_DELTAR_0->SetBinContent(29,1801.3001029600296);
  S42_DELTAR_0->SetBinContent(30,600.4332343200023);
  S42_DELTAR_0->SetBinContent(31,300.21661716000114);
  S42_DELTAR_0->SetBinContent(32,300.21661716000114);
  S42_DELTAR_0->SetBinContent(33,0.0);
  S42_DELTAR_0->SetBinContent(34,0.0);
  S42_DELTAR_0->SetBinContent(35,0.0);
  S42_DELTAR_0->SetBinContent(36,0.0);
  S42_DELTAR_0->SetBinContent(37,0.0);
  S42_DELTAR_0->SetBinContent(38,0.0);
  S42_DELTAR_0->SetBinContent(39,0.0);
  S42_DELTAR_0->SetBinContent(40,0.0);
  S42_DELTAR_0->SetBinContent(41,0.0); // overflow
  S42_DELTAR_0->SetEntries(10000);
  // Style
  S42_DELTAR_0->SetLineColor(9);
  S42_DELTAR_0->SetLineStyle(1);
  S42_DELTAR_0->SetLineWidth(1);
  S42_DELTAR_0->SetFillColor(9);
  S42_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_84","mystack");
  stack->Add(S42_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ l+_{1}, p_{3} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_41.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_41.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_41.eps");

}
