void selection_19()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo39","canvas_plotflow_tempo39",0,0,700,500);
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
  TH1F* S20_M_0 = new TH1F("S20_M_0","S20_M_0",40,0.0,500.0);
  // Content
  S20_M_0->SetBinContent(0,0.0); // underflow
  S20_M_0->SetBinContent(1,60043.316460000206);
  S20_M_0->SetBinContent(2,160615.89053049943);
  S20_M_0->SetBinContent(3,262989.78449479747);
  S20_M_0->SetBinContent(4,343147.5797688996);
  S20_M_0->SetBinContent(5,343147.5797688996);
  S20_M_0->SetBinContent(6,293011.38272480114);
  S20_M_0->SetBinContent(7,241374.18576919768);
  S20_M_0->SetBinContent(8,181030.58932690122);
  S20_M_0->SetBinContent(9,157613.69070750143);
  S20_M_0->SetBinContent(10,123989.49268989783);
  S20_M_0->SetBinContent(11,105676.1937696029);
  S20_M_0->SetBinContent(12,82559.56513249999);
  S20_M_0->SetBinContent(13,73553.06566350008);
  S20_M_0->SetBinContent(14,75054.14557500025);
  S20_M_0->SetBinContent(15,49235.51709720031);
  S20_M_0->SetBinContent(16,45933.137291900144);
  S20_M_0->SetBinContent(17,39328.37768129982);
  S20_M_0->SetBinContent(18,31222.528159199894);
  S20_M_0->SetBinContent(19,25518.408495500145);
  S20_M_0->SetBinContent(20,20414.7287964);
  S20_M_0->SetBinContent(21,28520.578318499924);
  S20_M_0->SetBinContent(22,20414.7287964);
  S20_M_0->SetBinContent(23,18613.42890260002);
  S20_M_0->SetBinContent(24,16511.909026500234);
  S20_M_0->SetBinContent(25,13509.74920349987);
  S20_M_0->SetBinContent(26,12008.659292000277);
  S20_M_0->SetBinContent(27,12008.659292000277);
  S20_M_0->SetBinContent(28,13509.74920349987);
  S20_M_0->SetBinContent(29,11108.009345100287);
  S20_M_0->SetBinContent(30,8105.848522099981);
  S20_M_0->SetBinContent(31,10507.579380500096);
  S20_M_0->SetBinContent(32,9006.49746900003);
  S20_M_0->SetBinContent(33,6604.765610599975);
  S20_M_0->SetBinContent(34,5704.115663699984);
  S20_M_0->SetBinContent(35,5704.115663699984);
  S20_M_0->SetBinContent(36,4503.248734500015);
  S20_M_0->SetBinContent(37,4803.465716799993);
  S20_M_0->SetBinContent(38,3602.5987876000245);
  S20_M_0->SetBinContent(39,6004.3316460000215);
  S20_M_0->SetBinContent(40,3902.8157699000017);
  S20_M_0->SetBinContent(41,72051.9857519999); // overflow
  S20_M_0->SetEntries(10000);
  // Style
  S20_M_0->SetLineColor(9);
  S20_M_0->SetLineStyle(1);
  S20_M_0->SetLineWidth(1);
  S20_M_0->SetFillColor(9);
  S20_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_40","mystack");
  stack->Add(S20_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_19.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_19.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_19.eps");

}
