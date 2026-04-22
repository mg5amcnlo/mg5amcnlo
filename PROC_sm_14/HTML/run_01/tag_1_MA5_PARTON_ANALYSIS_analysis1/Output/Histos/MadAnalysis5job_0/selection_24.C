void selection_24()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo49","canvas_plotflow_tempo49",0,0,700,500);
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
  TH1F* S25_M_0 = new TH1F("S25_M_0","S25_M_0",40,0.0,500.0);
  // Content
  S25_M_0->SetBinContent(0,0.0); // underflow
  S25_M_0->SetBinContent(1,0.0);
  S25_M_0->SetBinContent(2,0.0);
  S25_M_0->SetBinContent(3,0.0);
  S25_M_0->SetBinContent(4,0.0);
  S25_M_0->SetBinContent(5,0.0);
  S25_M_0->SetBinContent(6,5403.898677800008);
  S25_M_0->SetBinContent(7,10507.579373500097);
  S25_M_0->SetBinContent(8,15911.479051300044);
  S25_M_0->SetBinContent(9,18913.64887229982);
  S25_M_0->SetBinContent(10,24617.758532200158);
  S25_M_0->SetBinContent(11,35425.55788780005);
  S25_M_0->SetBinContent(12,57341.36658110024);
  S25_M_0->SetBinContent(13,78656.74531020023);
  S25_M_0->SetBinContent(14,102373.89389609802);
  S25_M_0->SetBinContent(15,110479.69341280092);
  S25_M_0->SetBinContent(16,115283.19312639888);
  S25_M_0->SetBinContent(17,119486.19287580083);
  S25_M_0->SetBinContent(18,113481.8932337989);
  S25_M_0->SetBinContent(19,121887.89273260279);
  S25_M_0->SetBinContent(20,113181.69325169791);
  S25_M_0->SetBinContent(21,95468.87430780027);
  S25_M_0->SetBinContent(22,98471.04412880004);
  S25_M_0->SetBinContent(23,93367.3644330999);
  S25_M_0->SetBinContent(24,91265.84455840012);
  S25_M_0->SetBinContent(25,95769.09428990008);
  S25_M_0->SetBinContent(26,88864.11470159993);
  S25_M_0->SetBinContent(27,75654.58548919986);
  S25_M_0->SetBinContent(28,81959.13511329981);
  S25_M_0->SetBinContent(29,63045.486240999984);
  S25_M_0->SetBinContent(30,57041.15659899985);
  S25_M_0->SetBinContent(31,59142.66647370022);
  S25_M_0->SetBinContent(32,58242.01652740023);
  S25_M_0->SetBinContent(33,51036.8169570003);
  S25_M_0->SetBinContent(34,53438.55681379988);
  S25_M_0->SetBinContent(35,44732.277332899765);
  S25_M_0->SetBinContent(36,43231.18742240018);
  S25_M_0->SetBinContent(37,43531.40740449998);
  S25_M_0->SetBinContent(38,39028.15767300002);
  S25_M_0->SetBinContent(39,37827.28774460023);
  S25_M_0->SetBinContent(40,30321.878192099906);
  S25_M_0->SetBinContent(41,657774.5607811005); // overflow
  S25_M_0->SetEntries(10000);
  // Style
  S25_M_0->SetLineColor(9);
  S25_M_0->SetLineStyle(1);
  S25_M_0->SetLineWidth(1);
  S25_M_0->SetFillColor(9);
  S25_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_50","mystack");
  stack->Add(S25_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} l-_{1} p_{1} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_24.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_24.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_24.eps");

}
