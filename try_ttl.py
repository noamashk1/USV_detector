import lgpio
import time

GPIO17 = 17  # BCM numbering
GPIO22 = 22  # BCM numbering
GPIO27 = 27  # BCM numbering

# Open GPIO chip (0 is correct for Raspberry Pi)
h = lgpio.gpiochip_open(0)

# Set GPIO18 as output
lgpio.gpio_claim_output(h, GPIO17)
lgpio.gpio_claim_output(h, GPIO22)
lgpio.gpio_claim_output(h, GPIO27)
GPIO_lst = [GPIO17,GPIO27, GPIO22]
idx = 0
try:
    while True:
        GPIO = GPIO_lst[idx%3]
        lgpio.gpio_write(h, GPIO, 1)  # HIGH (3.3V)
        print("playing..." + str(GPIO))
        time.sleep(0.5)               # pulse width
        lgpio.gpio_write(h, GPIO, 0)  # LOW
        time.sleep(3)               # rest of 3 sec period
        idx += 1        

except KeyboardInterrupt:
    pass

finally:
    lgpio.gpio_write(h, GPIO, 0)
    lgpio.gpiochip_close(h)

