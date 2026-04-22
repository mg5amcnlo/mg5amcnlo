void selection_38()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo77","canvas_plotflow_tempo77",0,0,700,500);
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
  TH1F* S39_M_0 = new TH1F("S39_M_0","S39_M_0",40,0.0,500.0);
  // Content
  S39_M_0->SetBinContent(0,0.0); // underflow
  S39_M_0->SetBinContent(1,0.0);
  S39_M_0->SetBinContent(2,146805.89525670072);
  S39_M_0->SetBinContent(3,316428.2897762002);
  S39_M_0->SetBinContent(4,368365.78809809935);
  S39_M_0->SetBinContent(5,334141.08920389955);
  S39_M_0->SetBinContent(6,270795.39125059947);
  S39_M_0->SetBinContent(7,218557.69293839976);
  S39_M_0->SetBinContent(8,175026.29434489945);
  S39_M_0->SetBinContent(9,139900.8954798013);
  S39_M_0->SetBinContent(10,119786.39612970088);
  S39_M_0->SetBinContent(11,103274.49666320045);
  S39_M_0->SetBinContent(12,83760.43729369981);
  S39_M_0->SetBinContent(13,76255.01753619997);
  S39_M_0->SetBinContent(14,57041.15815699987);
  S39_M_0->SetBinContent(15,58542.238108499965);
  S39_M_0->SetBinContent(16,41429.888661400066);
  S39_M_0->SetBinContent(17,42630.758622599955);
  S39_M_0->SetBinContent(18,41730.108651699964);
  S39_M_0->SetBinContent(19,30622.089010600135);
  S39_M_0->SetBinContent(20,30922.309000900026);
  S39_M_0->SetBinContent(21,27920.14909789983);
  S39_M_0->SetBinContent(22,26118.849156099837);
  S39_M_0->SetBinContent(23,18613.429398599994);
  S39_M_0->SetBinContent(24,23116.679253099966);
  S39_M_0->SetBinContent(25,13809.95955380013);
  S39_M_0->SetBinContent(26,12609.099592599923);
  S39_M_0->SetBinContent(27,12308.879602300032);
  S39_M_0->SetBinContent(28,6604.765786599981);
  S39_M_0->SetBinContent(29,15311.049505299905);
  S39_M_0->SetBinContent(30,13509.749563499918);
  S39_M_0->SetBinContent(31,9606.931689599985);
  S39_M_0->SetBinContent(32,10807.799650799932);
  S39_M_0->SetBinContent(33,7505.414757500008);
  S39_M_0->SetBinContent(34,8706.28171869999);
  S39_M_0->SetBinContent(35,10807.799650799932);
  S39_M_0->SetBinContent(36,5403.8988254);
  S39_M_0->SetBinContent(37,7205.198767199988);
  S39_M_0->SetBinContent(38,7805.631747799995);
  S39_M_0->SetBinContent(39,3902.8158738999973);
  S39_M_0->SetBinContent(40,5704.115815699986);
  S39_M_0->SetBinContent(41,98771.26680869983); // overflow
  S39_M_0->SetEntries(10000);
  // Style
  S39_M_0->SetLineColor(9);
  S39_M_0->SetLineStyle(1);
  S39_M_0->SetLineWidth(1);
  S39_M_0->SetFillColor(9);
  S39_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_78","mystack");
  stack->Add(S39_M_0);
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
  stack->GetXaxis()->SetTitle("M [ p_{2} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_38.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_38.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_38.eps");

}
