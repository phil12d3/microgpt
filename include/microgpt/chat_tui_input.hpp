#pragma once

#include <cstdio>
#include <utility>

#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace microgpt {

class RawTerminalGuard {
 public:
#if !defined(_WIN32)
  RawTerminalGuard() {
    if (tcgetattr(STDIN_FILENO, &original_) != 0) {
      active_ = false;
      return;
    }
    termios raw = original_;
    raw.c_lflag &= static_cast<unsigned>(~(ECHO | ICANON | IEXTEN | ISIG));
    raw.c_iflag &= static_cast<unsigned>(~(IXON | ICRNL));
    raw.c_oflag &= static_cast<unsigned>(~(OPOST));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
      active_ = false;
      return;
    }
    active_ = true;
    std::fputs("\033[?1049h\033[?25l", stdout);
    std::fflush(stdout);
  }

  ~RawTerminalGuard() {
    if (active_) {
      tcsetattr(STDIN_FILENO, TCSANOW, &original_);
      std::fputs("\033[?25h\033[?1049l", stdout);
      std::fflush(stdout);
    }
  }
#else
  RawTerminalGuard() = default;
  ~RawTerminalGuard() = default;
#endif

  bool active() const { return active_; }

 private:
#if !defined(_WIN32)
  termios original_{};
#endif
  bool active_ = false;
};

inline std::pair<int, int> terminal_size() {
#if !defined(_WIN32)
  winsize ws{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
    return {static_cast<int>(ws.ws_col), static_cast<int>(ws.ws_row)};
  }
#endif
  return {100, 30};
}

inline int read_key() {
#if !defined(_WIN32)
  unsigned char c = 0;
  ssize_t n = ::read(STDIN_FILENO, &c, 1);
  if (n <= 0) {
    return -1;
  }
  if (c == '\033') {
    unsigned char seq[2] = {0, 0};
    if (::read(STDIN_FILENO, &seq[0], 1) <= 0) {
      return 27;
    }
    if (::read(STDIN_FILENO, &seq[1], 1) <= 0) {
      return 27;
    }
    if (seq[0] == '[') {
      switch (seq[1]) {
        case 'A':
          return 1001;
        case 'B':
          return 1002;
        case 'C':
          return 1003;
        case 'D':
          return 1004;
        case 'H':
          return 1005;
        case 'F':
          return 1006;
      }
    }
    return 27;
  }
  return static_cast<int>(c);
#else
  return -1;
#endif
}

}  // namespace microgpt
