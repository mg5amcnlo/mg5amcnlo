void selection_46()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo93","canvas_plotflow_tempo93",0,0,700,500);
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
  TH1F* S47_DELTAR_0 = new TH1F("S47_DELTAR_0","S47_DELTAR_0",40,0.0,10.0);
  // Content
  S47_DELTAR_0->SetBinContent(0,0.0); // underflow
  S47_DELTAR_0->SetBinContent(1,7205.198388960005);
  S47_DELTAR_0->SetBinContent(2,43831.626282839796);
  S47_DELTAR_0->SetBinContent(3,63946.134577020144);
  S47_DELTAR_0->SetBinContent(4,80458.04317672053);
  S47_DELTAR_0->SetBinContent(5,92166.4921837804);
  S47_DELTAR_0->SetBinContent(6,83460.2129221202);
  S47_DELTAR_0->SetBinContent(7,113481.89037611874);
  S47_DELTAR_0->SetBinContent(8,135397.68851753994);
  S47_DELTAR_0->SetBinContent(9,142002.48795741703);
  S47_DELTAR_0->SetBinContent(10,196041.3833746249);
  S47_DELTAR_0->SetBinContent(11,268994.07718783984);
  S47_DELTAR_0->SetBinContent(12,323033.0726050392);
  S47_DELTAR_0->SetBinContent(13,366564.4689133402);
  S47_DELTAR_0->SetBinContent(14,243175.4793773973);
  S47_DELTAR_0->SetBinContent(15,175026.28515681947);
  S47_DELTAR_0->SetBinContent(16,164818.88602246242);
  S47_DELTAR_0->SetBinContent(17,113481.89037611874);
  S47_DELTAR_0->SetBinContent(18,90965.6222856207);
  S47_DELTAR_0->SetBinContent(19,76555.2335077);
  S47_DELTAR_0->SetBinContent(20,63645.91460248043);
  S47_DELTAR_0->SetBinContent(21,46833.786028240334);
  S47_DELTAR_0->SetBinContent(22,30622.08740308051);
  S47_DELTAR_0->SetBinContent(23,23717.107988660304);
  S47_DELTAR_0->SetBinContent(24,20114.50829418034);
  S47_DELTAR_0->SetBinContent(25,10507.579108900167);
  S47_DELTAR_0->SetBinContent(26,8105.848312579996);
  S47_DELTAR_0->SetBinContent(27,6004.3314908000475);
  S47_DELTAR_0->SetBinContent(28,3902.815669020013);
  S47_DELTAR_0->SetBinContent(29,2401.732796320002);
  S47_DELTAR_0->SetBinContent(30,2101.5158217800335);
  S47_DELTAR_0->SetBinContent(31,1801.2998472399802);
  S47_DELTAR_0->SetBinContent(32,600.4331490800047);
  S47_DELTAR_0->SetBinContent(33,900.6497236200071);
  S47_DELTAR_0->SetBinContent(34,0.0);
  S47_DELTAR_0->SetBinContent(35,0.0);
  S47_DELTAR_0->SetBinContent(36,0.0);
  S47_DELTAR_0->SetBinContent(37,0.0);
  S47_DELTAR_0->SetBinContent(38,300.21657454000234);
  S47_DELTAR_0->SetBinContent(39,0.0);
  S47_DELTAR_0->SetBinContent(40,0.0);
  S47_DELTAR_0->SetBinContent(41,0.0); // overflow
  S47_DELTAR_0->SetEntries(10000);
  // Style
  S47_DELTAR_0->SetLineColor(9);
  S47_DELTAR_0->SetLineStyle(1);
  S47_DELTAR_0->SetLineWidth(1);
  S47_DELTAR_0->SetFillColor(9);
  S47_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_94","mystack");
  stack->Add(S47_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ p_{1}, p_{2} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_46.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_46.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_46.eps");

}
