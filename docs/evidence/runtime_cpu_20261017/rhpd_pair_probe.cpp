#include <fstream>
#include <iostream>
#include <iomanip>
#include <ctime>
#include "n3mapping/RHPDescriptor.h"
using namespace n3mapping;
using D=RHPDescriptor;
double cpuMs(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
int main(int argc,char**argv){
 if(argc!=3)return 2;std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2]);
#ifdef N3M_SPARSE
 std::vector<RHPDPlace> places;std::vector<RHPDHeadingPtr> headings;
#else
 std::vector<Eigen::VectorXd> places;std::vector<Eigen::MatrixXd> headings;
#endif
 for(int sample=0;sample<240;++sample){Eigen::VectorXd place(2973);int mode=sample%4;Eigen::MatrixXd heading(mode==3?84:140,mode==3?72:60);in.read(reinterpret_cast<char*>(place.data()),place.size()*8);in.read(reinterpret_cast<char*>(heading.data()),heading.size()*8);if(!in)return 3;if(mode!=0)continue;
#ifdef N3M_SPARSE
  RHPDPlace p;for(int plane=0;plane<3;++plane)for(int cell=0;cell<196;++cell)p.planes[plane].append(cell,place.segment<4>(plane*784+cell*4));for(int cell=0;cell<128;++cell)p.ring_height.append(cell,place.segment<4>(2352+cell*4));p.auxiliary=place.tail(109);places.push_back(p);headings.push_back(makeRHPDHeading(heading));
#else
  places.push_back(place);headings.push_back(heading);
#endif
 }
 D descriptor(D::Params{});RHPDHeadingParams params;out<<std::setprecision(17)<<"query,reference,place_distance,heading_distance,shift\n";
 volatile double guard=0;
 for(int repeat=0;repeat<3;++repeat){double place_cpu=0,heading_cpu=0;
 for(size_t i=0;i<places.size();++i)for(size_t j=0;j<places.size();++j){auto start=cpuMs();double distance=descriptor.distance(places[i],places[j]);place_cpu+=cpuMs()-start;start=cpuMs();
#ifdef N3M_SPARSE
 auto h=matchRHPDHeading(*headings[i],*headings[j],params);
#else
 auto h=matchRHPDHeading(headings[i],headings[j],params);
#endif
 heading_cpu+=cpuMs()-start;guard+=distance+h.first;
 if(repeat==0)out<<i<<','<<j<<','<<distance<<','<<h.first<<','<<h.second<<'\n';
 }
 std::cout<<"repeat="<<repeat<<" pairs="<<places.size()*places.size()<<" place_cpu_ms="<<place_cpu<<" heading_cpu_ms="<<heading_cpu<<std::endl;
 }return 0;
}
