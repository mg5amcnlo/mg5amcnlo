void selection_43()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo87","canvas_plotflow_tempo87",0,0,700,500);
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
  TH1F* S44_DELTAR_0 = new TH1F("S44_DELTAR_0","S44_DELTAR_0",40,0.0,10.0);
  // Content
  S44_DELTAR_0->SetBinContent(0,0.0); // underflow
  S44_DELTAR_0->SetBinContent(1,11708.449792519958);
  S44_DELTAR_0->SetBinContent(2,27619.92951055996);
  S44_DELTAR_0->SetBinContent(3,49835.95911687994);
  S44_DELTAR_0->SetBinContent(4,66648.08881895994);
  S44_DELTAR_0->SetBinContent(5,87062.81845719993);
  S44_DELTAR_0->SetBinContent(6,103874.8981592808);
  S44_DELTAR_0->SetBinContent(7,123388.99781348044);
  S44_DELTAR_0->SetBinContent(8,168421.49701548027);
  S44_DELTAR_0->SetBinContent(9,194540.3965526393);
  S44_DELTAR_0->SetBinContent(10,262389.29535032023);
  S44_DELTAR_0->SetBinContent(11,322132.3942916403);
  S44_DELTAR_0->SetBinContent(12,406193.0928020394);
  S44_DELTAR_0->SetBinContent(13,413398.29267435934);
  S44_DELTAR_0->SetBinContent(14,223961.59603127977);
  S44_DELTAR_0->SetBinContent(15,141702.19748896066);
  S44_DELTAR_0->SetBinContent(16,104775.59814331992);
  S44_DELTAR_0->SetBinContent(17,80157.83857955989);
  S44_DELTAR_0->SetBinContent(18,64546.5688562);
  S44_DELTAR_0->SetBinContent(19,44432.05921263996);
  S44_DELTAR_0->SetBinContent(20,32123.17943075994);
  S44_DELTAR_0->SetBinContent(21,25218.19955311991);
  S44_DELTAR_0->SetBinContent(22,15010.829734000004);
  S44_DELTAR_0->SetBinContent(23,10207.359819120082);
  S44_DELTAR_0->SetBinContent(24,7505.414867000002);
  S44_DELTAR_0->SetBinContent(25,5704.115898919991);
  S44_DELTAR_0->SetBinContent(26,4203.032925519991);
  S44_DELTAR_0->SetBinContent(27,2701.94995211999);
  S44_DELTAR_0->SetBinContent(28,2101.515962760004);
  S44_DELTAR_0->SetBinContent(29,0.0);
  S44_DELTAR_0->SetBinContent(30,600.4331893600003);
  S44_DELTAR_0->SetBinContent(31,0.0);
  S44_DELTAR_0->SetBinContent(32,0.0);
  S44_DELTAR_0->SetBinContent(33,0.0);
  S44_DELTAR_0->SetBinContent(34,0.0);
  S44_DELTAR_0->SetBinContent(35,0.0);
  S44_DELTAR_0->SetBinContent(36,0.0);
  S44_DELTAR_0->SetBinContent(37,0.0);
  S44_DELTAR_0->SetBinContent(38,0.0);
  S44_DELTAR_0->SetBinContent(39,0.0);
  S44_DELTAR_0->SetBinContent(40,0.0);
  S44_DELTAR_0->SetBinContent(41,0.0); // overflow
  S44_DELTAR_0->SetEntries(10000);
  // Style
  S44_DELTAR_0->SetLineColor(9);
  S44_DELTAR_0->SetLineStyle(1);
  S44_DELTAR_0->SetLineWidth(1);
  S44_DELTAR_0->SetFillColor(9);
  S44_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_88","mystack");
  stack->Add(S44_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ l-_{1}, p_{1} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_43.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_43.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_43.eps");

}
