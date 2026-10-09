/* Timing harness (REPORT.md "Run 1"): build from the repo root with cost of ds4_session_mark_rewind_point and of a separate
 * 7-token sync at depth, Ornith with MTP (as production). Not committed. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
  int depth = argc>1?atoi(argv[1]):30000;
  ds4_engine_options opt={.model_path=getenv("DS4_TEST_MODEL"),.backend=DS4_BACKEND_METAL,.n_threads=1,.context_size=524288,.glm_mtp=true,.mtp_draft_tokens=1};
  ds4_engine*e=NULL; if(ds4_engine_open(&e,&opt)) return 1;
  FILE*f=fopen(argv[2],"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char*t=malloc(n+1); fread(t,1,n,f); t[n]=0;
  ds4_tokens all={0}; ds4_tokenize_text(e,t,&all);
  ds4_tokens a={0}; for(int i=0;i<depth;i++) ds4_tokens_push(&a, all.v[i % all.len]);
  char err[256]; ds4_session*s=NULL; ds4_session_create(&s,e,524288);
  for (int round=0; round<3; round++) {
    ds4_session_invalidate(s);
    ds4_tokens base={0}; for(int i=0;i<depth-7;i++) ds4_tokens_push(&base,a.v[i]);
    if(ds4_session_sync(s,&base,err,sizeof err)) {puts(err);return 1;}
    double t0=now(); bool ok=ds4_session_mark_rewind_point(s); double t1=now();
    if(ds4_session_sync(s,&a,err,sizeof err)) {puts(err);return 1;}
    double t2=now();
    /* control: same 7 tokens without a mark */
    ds4_session_invalidate(s);
    if(ds4_session_sync(s,&base,err,sizeof err)) {puts(err);return 1;}
    double t3=now();
    if(ds4_session_sync(s,&a,err,sizeof err)) {puts(err);return 1;}
    double t4=now();
    /* decode path: the same 7 tokens one eval at a time */
    ds4_session_invalidate(s);
    if(ds4_session_sync(s,&base,err,sizeof err)) {puts(err);return 1;}
    double t5=now();
    for (int i=depth-7;i<depth;i++) if(ds4_session_eval(s,a.v[i],err,sizeof err)) {puts(err);return 1;}
    double t6=now();
    printf("  eval x7 %.1f ms\n", (t6-t5)*1e3);
    printf("depth %d round %d: mark %s %.1f ms, 7-token sync after mark %.1f ms, 7-token sync no mark %.1f ms\n",
           depth, round, ok?"ok":"FAIL", (t1-t0)*1e3, (t2-t1)*1e3, (t4-t3)*1e3);
    ds4_tokens_free(&base);
  }
  return 0;
}
