// vkimp <device-name-substr> <file|anon> : import host memory into Vulkan, time GPU copies into a device-local buffer
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{VkResult r=(x);if(r!=VK_SUCCESS){printf("FAIL %s = %d line %d\n",#x,r,__LINE__);exit(1);}}while(0)
int main(int argc,char**argv){
  VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO};ai.apiVersion=VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&ai;
  VkInstance inst;CK(vkCreateInstance(&ici,0,&inst));
  uint32_t n=0;vkEnumeratePhysicalDevices(inst,&n,0);VkPhysicalDevice pd[16];vkEnumeratePhysicalDevices(inst,&n,pd);
  VkPhysicalDevice P=0;
  for(uint32_t i=0;i<n;i++){VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(pd[i],&p);printf("dev %u %s\n",i,p.deviceName);if(!P&&strstr(p.deviceName,argv[1]))P=pd[i];}
  if(!P){printf("no match\n");return 2;}
  const char*ext[]={"VK_EXT_external_memory_host"};
  float pr=1;VkDeviceQueueCreateInfo q={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(P,&nq,0);VkQueueFamilyProperties qp[16];vkGetPhysicalDeviceQueueFamilyProperties(P,&nq,qp);
  uint32_t qf=0;for(uint32_t i=0;i<nq;i++)if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qf=i;break;}
  q.queueFamilyIndex=qf;q.queueCount=1;q.pQueuePriorities=&pr;
  VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&q;dci.enabledExtensionCount=1;dci.ppEnabledExtensionNames=ext;
  VkDevice D;CK(vkCreateDevice(P,&dci,0,&D));
  VkQueue Q;vkGetDeviceQueue(D,qf,0,&Q);
  PFN_vkGetMemoryHostPointerPropertiesEXT getprops=(void*)vkGetDeviceProcAddr(D,"vkGetMemoryHostPointerPropertiesEXT");
  VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(P,&mp);
  size_t SZ=(size_t)256<<20, CH=1769472; 
  void*host;
  if(!strcmp(argv[2],"file")){int fd=open("/mnt/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf",O_RDONLY);host=mmap(0,SZ,PROT_READ,MAP_SHARED,fd,(off_t)4096*100000);if(host==MAP_FAILED){perror("mmap");return 3;}}
  else if(!strcmp(argv[2],"filepriv")){int fd=open("/mnt/models/lfm25/LFM2.5-24B-A2B-QAD-Q4_0.gguf",O_RDONLY);host=mmap(0,SZ,PROT_READ,MAP_PRIVATE,fd,(off_t)4096*100000);if(host==MAP_FAILED){perror("mmap");return 3;}}
  else {host=mmap(0,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);memset(host,3,SZ);}
  madvise(host,SZ,MADV_POPULATE_READ);
  VkMemoryHostPointerPropertiesEXT hp={VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
  CK(getprops(D,VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,host,&hp));
  printf("host pointer memoryTypeBits=%x\n",hp.memoryTypeBits);
  VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=SZ;bi.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
  VkExternalMemoryBufferCreateInfo ebi={VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};ebi.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;bi.pNext=&ebi;
  VkBuffer src;CK(vkCreateBuffer(D,&bi,0,&src));
  VkMemoryRequirements mr;vkGetBufferMemoryRequirements(D,src,&mr);
  uint32_t bits=mr.memoryTypeBits&hp.memoryTypeBits;uint32_t mt=0;for(;mt<32;mt++)if(bits&(1u<<mt))break;
  VkImportMemoryHostPointerInfoEXT imp={VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};imp.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;imp.pHostPointer=host;
  VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};mai.pNext=&imp;mai.allocationSize=SZ;mai.memoryTypeIndex=mt;
  double t0=now();VkDeviceMemory M;CK(vkAllocateMemory(D,&mai,0,&M));CK(vkBindBufferMemory(D,src,M,0));
  printf("import ok type %u flags %x in %.1f ms\n",mt,mp.memoryTypes[mt].propertyFlags,(now()-t0)*1e3);
  bi.pNext=0;bi.size=64<<20;bi.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;VkBuffer dst;CK(vkCreateBuffer(D,&bi,0,&dst));
  vkGetBufferMemoryRequirements(D,dst,&mr);
  uint32_t dt=0;for(;dt<mp.memoryTypeCount;dt++)if((mr.memoryTypeBits&(1u<<dt))&&(mp.memoryTypes[dt].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))break;
  mai.pNext=0;mai.allocationSize=mr.size;mai.memoryTypeIndex=dt;VkDeviceMemory DM;CK(vkAllocateMemory(D,&mai,0,&DM));CK(vkBindBufferMemory(D,dst,DM,0));
  VkCommandPoolCreateInfo cpi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=qf;VkCommandPool cp;CK(vkCreateCommandPool(D,&cpi,0,&cp));
  VkCommandBufferAllocateInfo cai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=cp;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;VkCommandBuffer cb;CK(vkAllocateCommandBuffers(D,&cai,&cb));
  VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence f;CK(vkCreateFence(D,&fi,0,&f));
  for(int pass=0;pass<3;pass++){
    int reps=100;t0=now();
    for(int r=0;r<reps;r++){
      VkCommandBufferBeginInfo bb={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bb.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkResetCommandBuffer(cb,0);CK(vkBeginCommandBuffer(cb,&bb));
      for(int c=0;c<3;c++){VkBufferCopy cr={(rand()%((SZ-CH*3)/4096))*4096ull,c*CH,CH};vkCmdCopyBuffer(cb,src,dst,1,&cr);}
      CK(vkEndCommandBuffer(cb));
      VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cb;
      CK(vkQueueSubmit(Q,1,&si,f));CK(vkWaitForFences(D,1,&f,1,~0ull));vkResetFences(D,1,&f);
    }
    double dt_=now()-t0;printf("pass %d: %d x 5.3 MB GPU copies from imported host memory: %.2f ms each, %.2f GB/s\n",pass,reps,dt_/reps*1e3,reps*3.0*CH/dt_/1e9);
  }
  return 0;}
