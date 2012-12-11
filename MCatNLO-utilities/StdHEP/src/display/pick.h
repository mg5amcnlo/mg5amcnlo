/* SCCS ID: pick.h 1.1 4/6/92 */
#define NO_TRACK -1

int FindTrack(StdHepWindow *window, int x, int y);
void ShowSelectedTrack(StdHepWindow *window);
void SelectTrack(StdHepWindow *window, int particleIndex);
void ShowTrackStats(StdHepWindow *window, int trackNum);
