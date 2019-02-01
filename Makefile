LD				= nvcc 
NVCC			= nvcc
SRC_DIR			= src
OBJ_DIR			= obj
BIN_DIR			= bin
NVFLAGS			= -Isrc -std=c++11 -O3
LDFLAGS			= -O3
TARGET			= main

CU_FILES		= $(shell find $(SRC_DIR)/*.cu)
H_FILES			= $(shell find $(SRC_DIR)/*.h)

CUO_FILES		= $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.cu.o)))

$(BIN_DIR)/$(TARGET) : $(OBJ_FILES) $(CUO_FILES)
		mkdir -p $(BIN_DIR)
		$(LD) $(LDFLAGS) -o $@ $?

$(CUO_FILES) : $(CU_FILES) $(CUH_FILES)
		mkdir -p $(OBJ_DIR)
		$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -r $(BIN_DIR) || true
	rm -r $(OBJ_DIR) || true