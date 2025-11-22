#include "mxc_device.h"
#include "mxc_delay.h"
#include "gpio.h"
#include <stdio.h>

// ==============================================================================
// Configuration
// ==============================================================================
#define NUM_LEDS 8
#define DELAY_MS 150
#define DELAY_WAVE 150
#define DELAY_EXPLODE 150

// Use SWITCH instead of BUTTON
#define SWITCH_PORT MXC_GPIO2
#define SWITCH_PIN  MXC_GPIO_PIN_3
#define SWITCH_MASK (MXC_GPIO_PIN_3)

// ==============================================================================
// LED pin mapping
// ==============================================================================
typedef struct {
    mxc_gpio_regs_t *port;
    uint32_t mask;
} gpio_map_t;

const gpio_map_t led_pins[NUM_LEDS] = {
    { MXC_GPIO1, MXC_GPIO_PIN_6  },
    { MXC_GPIO0, MXC_GPIO_PIN_9  },
    { MXC_GPIO0, MXC_GPIO_PIN_8  },
    { MXC_GPIO0, MXC_GPIO_PIN_11 },
    { MXC_GPIO0, MXC_GPIO_PIN_19 },
    { MXC_GPIO3, MXC_GPIO_PIN_1  },
    { MXC_GPIO0, MXC_GPIO_PIN_16 },
    { MXC_GPIO0, MXC_GPIO_PIN_17 }
};

// ==============================================================================
// Global Sequence State (Moved out of functions to allow resetting)
// ==============================================================================
// Sequence 1 State
int pp_pos = 0;
int pp_dir = 1;

// Sequence 2 State
int we_state = 0;
int we_i = 0;

// ==============================================================================
// Function Prototypes
// ==============================================================================
void init_leds(void);
void init_switch(void);
int  read_switch(void);
void update_leds(uint8_t pattern);

// returns 1 if sequence finished normally, 0 if interrupted by switch change
int delay_check_switch(uint32_t delay_ms, int current_switch_state); 

void reset_sequences(void);
void seq_pingpong_double_step(int current_switch_state);
void seq_wave_explode_step(int current_switch_state);

// ==============================================================================
// MAIN
// ==============================================================================
int main(void)
{
    printf("MAX78000 LED Sequencer with Real-Time Switch Control\n");

    init_leds();
    init_switch();
    
    int current_switch = read_switch();
    int last_switch = -1;

    while (1) {
        // check switch state at start of loop
        current_switch = read_switch();

        // ff switch changed, reset the sequence states so the new pattern starts fresh
        if (current_switch != last_switch) {
            reset_sequences();
            last_switch = current_switch;
        }

        if (current_switch == 0)
            seq_pingpong_double_step(current_switch);
        else
            seq_wave_explode_step(current_switch);
    }

    return 0;
}

// ==============================================================================
// GPIO Initialization
// ==============================================================================
void init_leds(void)
{
    mxc_gpio_cfg_t cfg;

    for (int i = 0; i < NUM_LEDS; i++) {
        cfg.port  = led_pins[i].port;
        cfg.mask  = led_pins[i].mask;
        cfg.func  = MXC_GPIO_FUNC_OUT;
        cfg.pad   = MXC_GPIO_PAD_NONE;
        cfg.vssel = MXC_GPIO_VSSEL_VDDIOH;

        MXC_GPIO_Config(&cfg);
        MXC_GPIO_OutClr(led_pins[i].port, led_pins[i].mask);
    }
}

void init_switch(void)
{
    mxc_gpio_cfg_t sw;
    sw.port  = SWITCH_PORT;
    sw.mask  = SWITCH_MASK;
    sw.func  = MXC_GPIO_FUNC_IN;
    sw.pad   = MXC_GPIO_PAD_PULL_UP;   // switch to GND
    sw.vssel = MXC_GPIO_VSSEL_VDDIO;

    MXC_GPIO_Config(&sw);
}

// ==============================================================================
// SWITCH READ
// ==============================================================================
int read_switch(void)
{
    // Active LOW switch: S = 0 when grounded, S = 1 when high
    return (SWITCH_PORT->in & SWITCH_MASK) ? 1 : 0;
}

// ==============================================================================
// INTERRUPTIBLE DELAY
// ==============================================================================

int delay_check_switch(uint32_t delay_ms, int expected_switch_state)
{
    for (uint32_t i = 0; i < delay_ms; i++) {
        MXC_Delay(MXC_DELAY_MSEC(1));
        
        if (read_switch() != expected_switch_state) {
            return 0; // switch changed!
        }
    }
    return 1; // Completed delay normally
}

// ==============================================================================
// LED OUTPUT
// ==============================================================================
void update_leds(uint8_t pattern)
{
    for (int i = 0; i < NUM_LEDS; i++) {
        if (pattern & (1 << i))
            MXC_GPIO_OutSet(led_pins[i].port, led_pins[i].mask);
        else
            MXC_GPIO_OutClr(led_pins[i].port, led_pins[i].mask);
    }
}

// ==============================================================================
// SEQUENCE LOGIC
// ==============================================================================

void reset_sequences(void)
{
    // reset all static counters to 0
    pp_pos = 0;
    pp_dir = 1;
    we_state = 0;
    we_i = 0;
    
    // clear LEDs for a clean visual transition
    update_leds(0x00);
}

// SEQUENCE 1: Ping-Pong Double
void seq_pingpong_double_step(int current_switch_state)
{
    int width = 2;
    uint8_t pattern = (0x03 << pp_pos);

    update_leds(pattern);

    if (!delay_check_switch(DELAY_MS, current_switch_state)) return;

    pp_pos += pp_dir;

    if (pp_pos >= NUM_LEDS - width) {
        pp_dir = -1;
        pp_pos = NUM_LEDS - width;
    }
    else if (pp_pos <= 0) {
        pp_dir = 1;
        pp_pos = 0;
    }
}

// SEQUENCE 2: Wave + Explode
void seq_wave_explode_step(int current_switch_state)
{
    
    int delay_time = DELAY_WAVE;

    switch (we_state) {

        case 0: // Wave L → R
            update_leds(1 << we_i);
            if (!delay_check_switch(DELAY_WAVE, current_switch_state)) return;
            
            we_i++;
            if (we_i >= NUM_LEDS) { we_i = NUM_LEDS - 2; we_state = 1; }
            break;

        case 1: // Wave R → L
            update_leds(1 << we_i);
            if (!delay_check_switch(DELAY_WAVE, current_switch_state)) return;
            
            we_i--;
            if (we_i <= -1) { we_state = 2; we_i = 0; }
            break;

        case 2: // Explode Outward
            update_leds((1 << (3 - we_i)) | (1 << (4 + we_i)));
            if (!delay_check_switch(DELAY_EXPLODE, current_switch_state)) return;
            
            we_i++;
            if (we_i >= 4) { we_i = 2; we_state = 3; }
            break;

        case 3: // Collapse Inward
            update_leds((1 << (3 - we_i)) | (1 << (4 + we_i)));
            if (!delay_check_switch(DELAY_EXPLODE, current_switch_state)) return;
            
            we_i--;
            if (we_i < 0) { we_i = 0; we_state = 0; }
            break;
    }
}