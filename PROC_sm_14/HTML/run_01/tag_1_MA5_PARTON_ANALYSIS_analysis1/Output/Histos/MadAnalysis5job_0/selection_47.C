void selection_47()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo95","canvas_plotflow_tempo95",0,0,700,500);
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
  TH1F* S48_DELTAR_0 = new TH1F("S48_DELTAR_0","S48_DELTAR_0",40,0.0,10.0);
  // Content
  S48_DELTAR_0->SetBinContent(0,0.0); // underflow
  S48_DELTAR_0->SetBinContent(1,1501.0828908000076);
  S48_DELTAR_0->SetBinContent(2,15911.478842480066);
  S48_DELTAR_0->SetBinContent(3,52838.11615616039);
  S48_DELTAR_0->SetBinContent(4,95468.8730548804);
  S48_DELTAR_0->SetBinContent(5,117384.6914605599);
  S48_DELTAR_0->SetBinContent(6,137799.3899754421);
  S48_DELTAR_0->SetBinContent(7,163017.58814088182);
  S48_DELTAR_0->SetBinContent(8,159414.98840296187);
  S48_DELTAR_0->SetBinContent(9,198142.9855855978);
  S48_DELTAR_0->SetBinContent(10,232667.88307399862);
  S48_DELTAR_0->SetBinContent(11,244076.08224408093);
  S48_DELTAR_0->SetBinContent(12,282804.07942671684);
  S48_DELTAR_0->SetBinContent(13,276199.2799071993);
  S48_DELTAR_0->SetBinContent(14,211952.8845809625);
  S48_DELTAR_0->SetBinContent(15,160315.68833743825);
  S48_DELTAR_0->SetBinContent(16,142302.68964783844);
  S48_DELTAR_0->SetBinContent(17,104475.39239967884);
  S48_DELTAR_0->SetBinContent(18,92166.49329512019);
  S48_DELTAR_0->SetBinContent(19,69049.8149768002);
  S48_DELTAR_0->SetBinContent(20,60943.96556648029);
  S48_DELTAR_0->SetBinContent(21,50136.166352720415);
  S48_DELTAR_0->SetBinContent(22,31522.73770680038);
  S48_DELTAR_0->SetBinContent(23,25818.628121759957);
  S48_DELTAR_0->SetBinContent(24,18012.9986895998);
  S48_DELTAR_0->SetBinContent(25,18913.648624079793);
  S48_DELTAR_0->SetBinContent(26,9606.93130111999);
  S48_DELTAR_0->SetBinContent(27,7805.63143216001);
  S48_DELTAR_0->SetBinContent(28,7505.414454000038);
  S48_DELTAR_0->SetBinContent(29,6004.33156320003);
  S48_DELTAR_0->SetBinContent(30,3002.165781600015);
  S48_DELTAR_0->SetBinContent(31,3002.165781600015);
  S48_DELTAR_0->SetBinContent(32,600.433156320003);
  S48_DELTAR_0->SetBinContent(33,1200.865912640035);
  S48_DELTAR_0->SetBinContent(34,300.2165781600015);
  S48_DELTAR_0->SetBinContent(35,0.0);
  S48_DELTAR_0->SetBinContent(36,300.2165781600015);
  S48_DELTAR_0->SetBinContent(37,0.0);
  S48_DELTAR_0->SetBinContent(38,0.0);
  S48_DELTAR_0->SetBinContent(39,0.0);
  S48_DELTAR_0->SetBinContent(40,0.0);
  S48_DELTAR_0->SetBinContent(41,0.0); // overflow
  S48_DELTAR_0->SetEntries(10000);
  // Style
  S48_DELTAR_0->SetLineColor(9);
  S48_DELTAR_0->SetLineStyle(1);
  S48_DELTAR_0->SetLineWidth(1);
  S48_DELTAR_0->SetFillColor(9);
  S48_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_96","mystack");
  stack->Add(S48_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ p_{1}, p_{3} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_47.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_47.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_47.eps");

}
