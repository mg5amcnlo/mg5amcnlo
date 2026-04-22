void selection_33()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo67","canvas_plotflow_tempo67",0,0,700,500);
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
  TH1F* S34_M_0 = new TH1F("S34_M_0","S34_M_0",40,0.0,500.0);
  // Content
  S34_M_0->SetBinContent(0,0.0); // underflow
  S34_M_0->SetBinContent(1,0.0);
  S34_M_0->SetBinContent(2,1200.8660008000002);
  S34_M_0->SetBinContent(3,6004.332004000003);
  S34_M_0->SetBinContent(4,31822.960021200008);
  S34_M_0->SetBinContent(5,70851.12004720002);
  S34_M_0->SetBinContent(6,116484.00007760001);
  S34_M_0->SetBinContent(7,148307.00009880005);
  S34_M_0->SetBinContent(8,166920.40011120003);
  S34_M_0->SetBinContent(9,168421.50011220004);
  S34_M_0->SetBinContent(10,170222.80011340004);
  S34_M_0->SetBinContent(11,165119.10011000003);
  S34_M_0->SetBinContent(12,144404.20009620007);
  S34_M_0->SetBinContent(13,137199.00009140006);
  S34_M_0->SetBinContent(14,128492.70008560004);
  S34_M_0->SetBinContent(15,115583.40007700004);
  S34_M_0->SetBinContent(16,110179.50007340004);
  S34_M_0->SetBinContent(17,84961.30005660003);
  S34_M_0->SetBinContent(18,81358.70005420002);
  S34_M_0->SetBinContent(19,82259.35005480003);
  S34_M_0->SetBinContent(20,72051.99004800004);
  S34_M_0->SetBinContent(21,61544.410041000025);
  S34_M_0->SetBinContent(22,59743.11003980003);
  S34_M_0->SetBinContent(23,61844.620041200025);
  S34_M_0->SetBinContent(24,55540.07003700002);
  S34_M_0->SetBinContent(25,51036.82003400001);
  S34_M_0->SetBinContent(26,39628.59002640001);
  S34_M_0->SetBinContent(27,38727.94002580002);
  S34_M_0->SetBinContent(28,37827.29002520001);
  S34_M_0->SetBinContent(29,31222.53002080001);
  S34_M_0->SetBinContent(30,34825.130023200014);
  S34_M_0->SetBinContent(31,31822.960021200008);
  S34_M_0->SetBinContent(32,24317.55001620001);
  S34_M_0->SetBinContent(33,27619.93001840001);
  S34_M_0->SetBinContent(34,22516.25001500001);
  S34_M_0->SetBinContent(35,24917.98001660001);
  S34_M_0->SetBinContent(36,19213.860012800003);
  S34_M_0->SetBinContent(37,20114.510013400002);
  S34_M_0->SetBinContent(38,21615.60001440001);
  S34_M_0->SetBinContent(39,18913.65001260001);
  S34_M_0->SetBinContent(40,17712.780011800005);
  S34_M_0->SetBinContent(41,299616.20019960013); // overflow
  S34_M_0->SetEntries(10000);
  // Style
  S34_M_0->SetLineColor(9);
  S34_M_0->SetLineStyle(1);
  S34_M_0->SetLineWidth(1);
  S34_M_0->SetFillColor(9);
  S34_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_68","mystack");
  stack->Add(S34_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l-_{1} p_{2} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_33.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_33.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_33.eps");

}
