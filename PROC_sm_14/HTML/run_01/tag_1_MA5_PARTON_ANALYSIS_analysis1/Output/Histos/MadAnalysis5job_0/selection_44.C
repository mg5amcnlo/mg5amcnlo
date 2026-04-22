void selection_44()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo89","canvas_plotflow_tempo89",0,0,700,500);
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
  TH1F* S45_DELTAR_0 = new TH1F("S45_DELTAR_0","S45_DELTAR_0",40,0.0,10.0);
  // Content
  S45_DELTAR_0->SetBinContent(0,0.0); // underflow
  S45_DELTAR_0->SetBinContent(1,20714.94824601976);
  S45_DELTAR_0->SetBinContent(2,66047.6544075998);
  S45_DELTAR_0->SetBinContent(3,96369.52184018058);
  S45_DELTAR_0->SetBinContent(4,124890.08942528137);
  S45_DELTAR_0->SetBinContent(5,145304.78769672394);
  S45_DELTAR_0->SetBinContent(6,163618.0861460967);
  S45_DELTAR_0->SetBinContent(7,183732.5844429579);
  S45_DELTAR_0->SetBinContent(8,211352.48210432037);
  S45_DELTAR_0->SetBinContent(9,214654.8818246989);
  S45_DELTAR_0->SetBinContent(10,234168.98017239728);
  S45_DELTAR_0->SetBinContent(11,258186.27813879983);
  S45_DELTAR_0->SetBinContent(12,275598.87666443683);
  S45_DELTAR_0->SetBinContent(13,258786.67808796265);
  S45_DELTAR_0->SetBinContent(14,168121.28576480085);
  S45_DELTAR_0->SetBinContent(15,128192.48914565993);
  S45_DELTAR_0->SetBinContent(16,103274.49125552163);
  S45_DELTAR_0->SetBinContent(17,75054.14364500053);
  S45_DELTAR_0->SetBinContent(18,65447.21445844037);
  S45_DELTAR_0->SetBinContent(19,47734.435958220296);
  S45_DELTAR_0->SetBinContent(20,42330.536415780356);
  S45_DELTAR_0->SetBinContent(21,28520.577585099953);
  S45_DELTAR_0->SetBinContent(22,28520.577585099953);
  S45_DELTAR_0->SetBinContent(23,19514.07834770006);
  S45_DELTAR_0->SetBinContent(24,15911.478652740097);
  S45_DELTAR_0->SetBinContent(25,11408.229034040149);
  S45_DELTAR_0->SetBinContent(26,3902.8156695400107);
  S45_DELTAR_0->SetBinContent(27,6604.765440759979);
  S45_DELTAR_0->SetBinContent(28,2401.7327966400003);
  S45_DELTAR_0->SetBinContent(29,1200.8658983200426);
  S45_DELTAR_0->SetBinContent(30,300.2165745800022);
  S45_DELTAR_0->SetBinContent(31,300.2165745800022);
  S45_DELTAR_0->SetBinContent(32,0.0);
  S45_DELTAR_0->SetBinContent(33,0.0);
  S45_DELTAR_0->SetBinContent(34,0.0);
  S45_DELTAR_0->SetBinContent(35,0.0);
  S45_DELTAR_0->SetBinContent(36,0.0);
  S45_DELTAR_0->SetBinContent(37,0.0);
  S45_DELTAR_0->SetBinContent(38,0.0);
  S45_DELTAR_0->SetBinContent(39,0.0);
  S45_DELTAR_0->SetBinContent(40,0.0);
  S45_DELTAR_0->SetBinContent(41,0.0); // overflow
  S45_DELTAR_0->SetEntries(10000);
  // Style
  S45_DELTAR_0->SetLineColor(9);
  S45_DELTAR_0->SetLineStyle(1);
  S45_DELTAR_0->SetLineWidth(1);
  S45_DELTAR_0->SetFillColor(9);
  S45_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_90","mystack");
  stack->Add(S45_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ l-_{1}, p_{2} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_44.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_44.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_44.eps");

}
