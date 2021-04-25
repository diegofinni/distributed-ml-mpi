TARGET = dml
OBJS += dml.o
OBJS += main.o

CC = mpicc
#CFLAGS = -Wall -Werror -DDEBUG -g
CFLAGS = -std=gnu99 -Wall -g -O3
CFLAGS += -MMD -MP
LDFLAGS += $(LIBS)

default:	$(TARGET)
all:		$(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

DEPS = $(OBJS:%.o=%.d)
-include $(DEPS)

clean:
	rm $(TARGET) $(OBJS) $(DEPS) || true