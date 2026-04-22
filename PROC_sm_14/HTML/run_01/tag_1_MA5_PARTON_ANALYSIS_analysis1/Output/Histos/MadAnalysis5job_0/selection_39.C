void selection_39()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo79","canvas_plotflow_tempo79",0,0,700,500);
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
  TH1F* S40_DELTAR_0 = new TH1F("S40_DELTAR_0","S40_DELTAR_0",40,0.0,10.0);
  // Content
  S40_DELTAR_0->SetBinContent(0,0.0); // underflow
  S40_DELTAR_0->SetBinContent(1,7205.1988267199895);
  S40_DELTAR_0->SetBinContent(2,33924.47918413992);
  S40_DELTAR_0->SetBinContent(3,43531.408953099955);
  S40_DELTAR_0->SetBinContent(4,65447.21842604002);
  S40_DELTAR_0->SetBinContent(5,100272.29758852113);
  S40_DELTAR_0->SetBinContent(6,108378.19739357989);
  S40_DELTAR_0->SetBinContent(7,135097.49675099936);
  S40_DELTAR_0->SetBinContent(8,176527.39575463918);
  S40_DELTAR_0->SetBinContent(9,212253.09489546102);
  S40_DELTAR_0->SetBinContent(10,268693.89353809913);
  S40_DELTAR_0->SetBinContent(11,316428.2923901201);
  S40_DELTAR_0->SetBinContent(12,392383.09056346014);
  S40_DELTAR_0->SetBinContent(13,407994.3901880193);
  S40_DELTAR_0->SetBinContent(14,224561.99459944054);
  S40_DELTAR_0->SetBinContent(15,138099.59667880097);
  S40_DELTAR_0->SetBinContent(16,99972.12759574002);
  S40_DELTAR_0->SetBinContent(17,75654.58818055988);
  S40_DELTAR_0->SetBinContent(18,54939.63867873998);
  S40_DELTAR_0->SetBinContent(19,38427.7290758399);
  S40_DELTAR_0->SetBinContent(20,27019.499350199872);
  S40_DELTAR_0->SetBinContent(21,26118.849371859877);
  S40_DELTAR_0->SetBinContent(22,17112.349588459918);
  S40_DELTAR_0->SetBinContent(23,10207.359754520114);
  S40_DELTAR_0->SetBinContent(24,8706.28179061999);
  S40_DELTAR_0->SetBinContent(25,6604.765841159984);
  S40_DELTAR_0->SetBinContent(26,2401.732942239997);
  S40_DELTAR_0->SetBinContent(27,2401.732942239997);
  S40_DELTAR_0->SetBinContent(28,300.2165927800002);
  S40_DELTAR_0->SetBinContent(29,1200.8659711200103);
  S40_DELTAR_0->SetBinContent(30,300.2165927800002);
  S40_DELTAR_0->SetBinContent(31,0.0);
  S40_DELTAR_0->SetBinContent(32,0.0);
  S40_DELTAR_0->SetBinContent(33,0.0);
  S40_DELTAR_0->SetBinContent(34,0.0);
  S40_DELTAR_0->SetBinContent(35,0.0);
  S40_DELTAR_0->SetBinContent(36,0.0);
  S40_DELTAR_0->SetBinContent(37,0.0);
  S40_DELTAR_0->SetBinContent(38,0.0);
  S40_DELTAR_0->SetBinContent(39,0.0);
  S40_DELTAR_0->SetBinContent(40,0.0);
  S40_DELTAR_0->SetBinContent(41,0.0); // overflow
  S40_DELTAR_0->SetEntries(10000);
  // Style
  S40_DELTAR_0->SetLineColor(9);
  S40_DELTAR_0->SetLineStyle(1);
  S40_DELTAR_0->SetLineWidth(1);
  S40_DELTAR_0->SetFillColor(9);
  S40_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_80","mystack");
  stack->Add(S40_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ l+_{1}, p_{1} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_39.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_39.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_39.eps");

}
