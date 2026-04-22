void selection_14()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo29","canvas_plotflow_tempo29",0,0,700,500);
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
  TH1F* S15_M_0 = new TH1F("S15_M_0","S15_M_0",40,0.0,500.0);
  // Content
  S15_M_0->SetBinContent(0,0.0); // underflow
  S15_M_0->SetBinContent(1,0.0);
  S15_M_0->SetBinContent(2,0.0);
  S15_M_0->SetBinContent(3,2401.732984799999);
  S15_M_0->SetBinContent(4,4803.465969599998);
  S15_M_0->SetBinContent(5,17112.349891699978);
  S15_M_0->SetBinContent(6,35725.77977389997);
  S15_M_0->SetBinContent(7,67848.94957060002);
  S15_M_0->SetBinContent(8,82559.56947749999);
  S15_M_0->SetBinContent(9,101473.19935780008);
  S15_M_0->SetBinContent(10,112581.19928750017);
  S15_M_0->SetBinContent(11,114082.29927800006);
  S15_M_0->SetBinContent(12,132095.29916400003);
  S15_M_0->SetBinContent(13,125190.29920770015);
  S15_M_0->SetBinContent(14,118885.79924759985);
  S15_M_0->SetBinContent(15,119185.99924569995);
  S15_M_0->SetBinContent(16,105676.19933120029);
  S15_M_0->SetBinContent(17,101172.99935969997);
  S15_M_0->SetBinContent(18,90965.62942430002);
  S15_M_0->SetBinContent(19,96669.74938819998);
  S15_M_0->SetBinContent(20,87363.02944710001);
  S15_M_0->SetBinContent(21,75054.149525);
  S15_M_0->SetBinContent(22,87062.81944899997);
  S15_M_0->SetBinContent(23,81658.91948319998);
  S15_M_0->SetBinContent(24,67848.94957060002);
  S15_M_0->SetBinContent(25,63345.70959909996);
  S15_M_0->SetBinContent(26,59743.10962189997);
  S15_M_0->SetBinContent(27,54639.41965420001);
  S15_M_0->SetBinContent(28,59142.66962570001);
  S15_M_0->SetBinContent(29,46833.78970360001);
  S15_M_0->SetBinContent(30,40829.45974159999);
  S15_M_0->SetBinContent(31,37827.28976060002);
  S15_M_0->SetBinContent(32,39028.15975299999);
  S15_M_0->SetBinContent(33,40529.23974350001);
  S15_M_0->SetBinContent(34,43531.40972449999);
  S15_M_0->SetBinContent(35,32723.609792900003);
  S15_M_0->SetBinContent(36,37226.8597644);
  S15_M_0->SetBinContent(37,33624.259787200004);
  S15_M_0->SetBinContent(38,28520.579819499988);
  S15_M_0->SetBinContent(39,30021.659810000005);
  S15_M_0->SetBinContent(40,30622.089806200023);
  S15_M_0->SetBinContent(41,496558.29685739975); // overflow
  S15_M_0->SetEntries(10000);
  // Style
  S15_M_0->SetLineColor(9);
  S15_M_0->SetLineStyle(1);
  S15_M_0->SetLineWidth(1);
  S15_M_0->SetFillColor(9);
  S15_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_30","mystack");
  stack->Add(S15_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} p_{1} p_{2} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_14.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_14.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_14.eps");

}
