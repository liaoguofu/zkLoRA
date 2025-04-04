# #原无gd_to_AB
# NVCC := $(CONDA_PREFIX)/bin/nvcc

# # Include and library paths
# INCLUDES := -I$(CONDA_PREFIX)/include
# LIBS := -L$(CONDA_PREFIX)/lib  

# # Get compute capability
# ARCH := sm_80

# # NVCC compiler flags
# NVCC_FLAGS := -arch=$(ARCH) -std=c++17 -O2

# # Source and object files
# CU_SRCS := bls12-381.cu ioutils.cu commitment.cu fr-tensor.cu g1-tensor.cu proof.cu zkrelu.cu zkfc.cu tlookup.cu polynomial.cu zksoftmax.cu rescaling.cu 
# CU_OBJS := $(CU_SRCS:.cu=.o)
# CPP_SRCS := $(wildcard *.cpp)
# CPP_OBJS := $(CPP_SRCS:.cpp=.o)
# TARGETS := main ppgen commit-param self-attn ffn rmsnorm skip-connection cel gd_to_AB

# # Automatically generate dependency files
# %.d: %.cu
# 	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -M $< > $@.$$$$; \
# 	sed 's,\($*\)\.o[ :]*,\1.o \1.d : ,g' < $@.$$$$ > $@; \
# 	rm -f $@.$$$$

# %.d: %.cpp
# 	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -M $< > $@.$$$$; \
# 	sed 's,\($*\)\.o[ :]*,\1.o \1.d : ,g' < $@.$$$$ > $@; \
# 	rm -f $@.$$$$

# # Include dependency files
# -include $(CU_SRCS:.cu=.d) $(CPP_SRCS:.cpp=.d)

# # Pattern rule for CUDA source files
# %.o: %.cu
# 	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# # Pattern rule for C++ source files
# %.o: %.cpp
# 	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# # General rule for building each target
# $(TARGETS): % : %.o $(CU_OBJS) $(CPP_OBJS)
# 	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

# # Clean rule
# clean:
# 	rm -f $(CU_SRCS:.cu=.d) $(CPP_SRCS:.cpp=.d) $(CU_OBJS) $(CPP_OBJS) $(TARGETS)

# # Default rule
# all: $(TARGETS)


# 禁用所有内置隐式规则（关键第一步）
MAKEFLAGS += -r

# 编译器配置
NVCC := $(CONDA_PREFIX)/bin/nvcc
ARCH := sm_80
NVCC_FLAGS := -arch=$(ARCH) -std=c++17 -O2 -Xcompiler "-Wno-unused-result"
INCLUDES := -I$(CONDA_PREFIX)/include
LIBS := -L$(CONDA_PREFIX)/lib

# 源文件定义
MAIN_SRCS := main.cu ppgen.cu cel.cu gd-to-AB.cu commit-param.cu self-attn.cu ffn.cu rmsnorm.cu skip-connection.cu 

COMMON_CU := bls12-381.cu ioutils.cu commitment.cu fr-tensor.cu \
            g1-tensor.cu proof.cu zkrelu.cu zkfc.cu tlookup.cu \
            polynomial.cu zksoftmax.cu rescaling.cu
CPP_SRCS  := timer.cpp

# 目标转换
ALL_TARGETS  := $(basename $(MAIN_SRCS))
COMMON_OBJS  := $(COMMON_CU:.cu=.o) $(CPP_SRCS:.cpp=.o)

# 显式声明伪目标
.PHONY: all clean

all: $(ALL_TARGETS)

# 单一静态模式规则（避免任何重复）
$(ALL_TARGETS): % : %.o $(COMMON_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@

# 编译规则（显式声明）
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

%.o: %.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc $< -o $@

# 唯一clean规则
clean:
	rm -f *.d *.o $(ALL_TARGETS)

# 依赖处理（确保不引入额外规则）
DEP_FILES := $(MAIN_SRCS:.cu=.d) $(COMMON_CU:.cu=.d) $(CPP_SRCS:.cpp=.d)

ifneq ($(MAKECMDGOALS),clean)
-include $(DEP_FILES)
endif

# 改进的依赖生成（避免规则污染）
%.d: %.cu
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -M $< | \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' > $@.tmp
	@mv $@.tmp $@

%.d: %.cpp
	@$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -M $< | \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' > $@.tmp
	@mv $@.tmp $@