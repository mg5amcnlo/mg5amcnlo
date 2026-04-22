void selection_17()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo35","canvas_plotflow_tempo35",0,0,700,500);
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
  TH1F* S18_M_0 = new TH1F("S18_M_0","S18_M_0",40,0.0,500.0);
  // Content
  S18_M_0->SetBinContent(0,0.0); // underflow
  S18_M_0->SetBinContent(1,54639.415286200514);
  S18_M_0->SetBinContent(2,120386.88961409716);
  S18_M_0->SetBinContent(3,178028.4846412965);
  S18_M_0->SetBinContent(4,266592.3770007969);
  S18_M_0->SetBinContent(5,277700.3760424982);
  S18_M_0->SetBinContent(6,292410.9747733995);
  S18_M_0->SetBinContent(7,261188.47746699696);
  S18_M_0->SetBinContent(8,200244.4827246991);
  S18_M_0->SetBinContent(9,174125.58497800372);
  S18_M_0->SetBinContent(10,143503.48761980407);
  S18_M_0->SetBinContent(11,126991.58904430285);
  S18_M_0->SetBinContent(12,103274.49109040167);
  S18_M_0->SetBinContent(13,86162.16256670015);
  S18_M_0->SetBinContent(14,74753.9335509);
  S18_M_0->SetBinContent(15,65447.214353800395);
  S18_M_0->SetBinContent(16,55540.065208500506);
  S18_M_0->SetBinContent(17,43831.62621859978);
  S18_M_0->SetBinContent(18,42030.3263739998);
  S18_M_0->SetBinContent(19,51637.25554519998);
  S18_M_0->SetBinContent(20,37827.28673660042);
  S18_M_0->SetBinContent(21,30622.087358200508);
  S18_M_0->SetBinContent(22,29121.007487700237);
  S18_M_0->SetBinContent(23,27019.497668999687);
  S18_M_0->SetBinContent(24,20714.94821289976);
  S18_M_0->SetBinContent(25,21015.158187000332);
  S18_M_0->SetBinContent(26,15911.478627300103);
  S18_M_0->SetBinContent(27,13209.528860400134);
  S18_M_0->SetBinContent(28,15611.258653200393);
  S18_M_0->SetBinContent(29,11708.448989899865);
  S18_M_0->SetBinContent(30,13509.748834499844);
  S18_M_0->SetBinContent(31,12909.308886300425);
  S18_M_0->SetBinContent(32,8406.064274800046);
  S18_M_0->SetBinContent(33,8406.064274800046);
  S18_M_0->SetBinContent(34,6004.331482000046);
  S18_M_0->SetBinContent(35,8706.281248900013);
  S18_M_0->SetBinContent(36,6304.548456100013);
  S18_M_0->SetBinContent(37,5704.115507899991);
  S18_M_0->SetBinContent(38,6004.331482000046);
  S18_M_0->SetBinContent(39,5103.681559700056);
  S18_M_0->SetBinContent(40,3902.815663300012);
  S18_M_0->SetBinContent(41,75954.79344730056); // overflow
  S18_M_0->SetEntries(10000);
  // Style
  S18_M_0->SetLineColor(9);
  S18_M_0->SetLineStyle(1);
  S18_M_0->SetLineWidth(1);
  S18_M_0->SetFillColor(9);
  S18_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_36","mystack");
  stack->Add(S18_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} p_{2} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_17.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_17.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_17.eps");

}
