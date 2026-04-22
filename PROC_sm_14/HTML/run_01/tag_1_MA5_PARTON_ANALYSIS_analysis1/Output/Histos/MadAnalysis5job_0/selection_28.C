void selection_28()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo57","canvas_plotflow_tempo57",0,0,700,500);
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
  TH1F* S29_M_0 = new TH1F("S29_M_0","S29_M_0",40,0.0,500.0);
  // Content
  S29_M_0->SetBinContent(0,0.0); // underflow
  S29_M_0->SetBinContent(1,21615.599459999896);
  S29_M_0->SetBinContent(2,57041.1585749999);
  S29_M_0->SetBinContent(3,95769.09760749998);
  S29_M_0->SetBinContent(4,157913.89605500092);
  S29_M_0->SetBinContent(5,173224.99567249962);
  S29_M_0->SetBinContent(6,216155.994599999);
  S29_M_0->SetBinContent(7,190637.4952375012);
  S29_M_0->SetBinContent(8,199643.99501250114);
  S29_M_0->SetBinContent(9,200544.6949899999);
  S29_M_0->SetBinContent(10,179229.29552250038);
  S29_M_0->SetBinContent(11,144404.19639249975);
  S29_M_0->SetBinContent(12,138700.09653499935);
  S29_M_0->SetBinContent(13,120987.29697749985);
  S29_M_0->SetBinContent(14,98771.26753249987);
  S29_M_0->SetBinContent(15,98471.04753999996);
  S29_M_0->SetBinContent(16,81959.13795249986);
  S29_M_0->SetBinContent(17,75654.5881099999);
  S29_M_0->SetBinContent(18,71751.7682075);
  S29_M_0->SetBinContent(19,63045.48842499995);
  S29_M_0->SetBinContent(20,56140.5085974999);
  S29_M_0->SetBinContent(21,50136.168747500094);
  S29_M_0->SetBinContent(22,47134.00882249995);
  S29_M_0->SetBinContent(23,38427.729039999904);
  S29_M_0->SetBinContent(24,35425.559115);
  S29_M_0->SetBinContent(25,31522.739212500102);
  S29_M_0->SetBinContent(26,30922.309227500024);
  S29_M_0->SetBinContent(27,22816.459430000057);
  S29_M_0->SetBinContent(28,24617.75938500005);
  S29_M_0->SetBinContent(29,23416.89941499989);
  S29_M_0->SetBinContent(30,23416.89941499989);
  S29_M_0->SetBinContent(31,13509.749662499937);
  S29_M_0->SetBinContent(32,17112.34957249992);
  S29_M_0->SetBinContent(33,12909.309677500105);
  S29_M_0->SetBinContent(34,15010.829625000013);
  S29_M_0->SetBinContent(35,9306.714767499998);
  S29_M_0->SetBinContent(36,11108.009722500115);
  S29_M_0->SetBinContent(37,10807.799729999948);
  S29_M_0->SetBinContent(38,9306.714767499998);
  S29_M_0->SetBinContent(39,10207.35974500012);
  S29_M_0->SetBinContent(40,7505.4148125000065);
  S29_M_0->SetBinContent(41,115883.5971050003); // overflow
  S29_M_0->SetEntries(10000);
  // Style
  S29_M_0->SetLineColor(9);
  S29_M_0->SetLineStyle(1);
  S29_M_0->SetLineWidth(1);
  S29_M_0->SetFillColor(9);
  S29_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_58","mystack");
  stack->Add(S29_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l-_{1} p_{1} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_28.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_28.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_28.eps");

}
