#include <fstream>
#include <iostream>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include "n3mapping/RHPDescriptor.h"
#ifndef N3M_FUSED
#include "Scancontext/HybridScancontext.h"
#endif
using D = n3mapping::RHPDescriptor;
double cpuMs() { timespec t{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t); return t.tv_sec*1000.0+t.tv_nsec/1e6; }
int main(int argc,char** argv) {
 if(argc<4)return 2;
 std::ofstream out(argv[1],std::ios::binary);
#ifdef N3M_FUSED
 std::ofstream storage(std::string(argv[1])+".csv");
 storage<<"sample,mode,place_values,place_bytes,heading_values,heading_bytes,dense_bytes\n";
#endif
 const int repetitions=std::stoi(argv[2]);double cpu=0;size_t samples=0;double guard=0;
 for(int arg=3;arg<argc;++arg) {
  D::CloudT::Ptr original(new D::CloudT);
  if(pcl::io::loadPCDFile(argv[arg],*original))return 3;
  for(int yaw : {0,42,137,180}) {
   D::CloudT::Ptr cloud(new D::CloudT(*original));
   const double a=yaw*M_PI/180.0;
   for(auto& p:*cloud){const double x=p.x,y=p.y;p.x=std::cos(a)*x-std::sin(a)*y;p.y=std::sin(a)*x+std::cos(a)*y;}
   for(int mode=0;mode<4;++mode) {
    D::Params p;p.v2_enable=mode!=1;p.v3_enable=mode==2;
    if(mode==3){p.enable_negative_space=false;p.enable_pca_confidence=false;p.max_range=17.0;p.z_min=-1.5;p.z_max=4.0;}
#ifdef N3M_FUSED
    if(mode==3){p.heading.rings=12;p.heading.sectors=72;p.heading.max_radius=35.0;}
#else
    if(mode==3) p.enable_vertical_tokens=false;
    HybridSCManager sc;
    if(mode==3){sc.PC_NUM_RING=12;sc.PC_NUM_SECTOR=72;sc.PC_MAX_RADIUS=35.0;}
#endif
    D descriptor(p);
#ifdef N3M_FUSED
    D::Place sparse_place;n3mapping::RHPDHeadingPtr sparse_heading;
#else
    D::VecD place;Eigen::MatrixXd heading;
#endif
    const double before=cpuMs();
    for(int i=0;i<repetitions;++i) {
#ifdef N3M_FUSED
     auto features=descriptor.computeWithHeading(cloud);sparse_place=std::move(features.place);sparse_heading=std::move(features.heading);
     guard+=sparse_place.auxiliary(108)+sparse_heading->sector_key(0,0);
#else
     place=descriptor.compute(cloud);heading=sc.makeScancontext(*cloud);
     guard+=place(place.size()-1)+heading.col(0).mean();
#endif

    }
    cpu+=cpuMs()-before;
#ifdef N3M_FUSED
    auto place=n3mapping::rhpdDenseValues(sparse_place);
    auto heading=n3mapping::rhpdHeadingDenseValues(*sparse_heading);
    size_t place_bytes=sizeof(sparse_place);
    const auto cell_bytes=[](const n3mapping::RHPDCells& c) { return c.indices.capacity()*sizeof(uint16_t)+c.values.capacity()*sizeof(Eigen::Vector4d); };
    for(const auto& plane:sparse_place.planes)place_bytes+=cell_bytes(plane);
    place_bytes+=cell_bytes(sparse_place.ring_height);
    const auto& h=*sparse_heading;
    size_t heading_bytes=sizeof(h)+h.offsets.capacity()*sizeof(int)+h.ring_indices.capacity()*sizeof(int)+h.values.capacity()*sizeof(double)+h.ring_key.capacity()*sizeof(float)+(h.column_norms.size()+h.sector_key.size())*sizeof(double);
    storage<<samples<<','<<mode<<','<<sparse_place.storedValues()<<','<<place_bytes<<','<<h.values.size()<<','<<heading_bytes<<','<<(place.size()+heading.size())*sizeof(double)<<'\n';
#else
    // Compare surviving channels; removed token slots are excluded only from the dump.
    D::VecD compact(2973);compact.head(2972)=place.head(2972);compact(2972)=place(place.size()-1);place=compact;
#endif
    out.write(reinterpret_cast<const char*>(place.data()),place.size()*sizeof(double));
    out.write(reinterpret_cast<const char*>(heading.data()),heading.size()*sizeof(double));++samples;
   }
  }
 }
 std::cout<<"samples="<<samples<<" repetitions="<<repetitions<<" thread_cpu_ms="<<cpu<<" guard="<<guard<<'\n';
}
