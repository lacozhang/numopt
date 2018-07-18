/*
 * =====================================================================================
 *
 *       Filename:  env.h
 *
 *    Description:  environment variable
 *
 *        Version:  1.0
 *        Created:  07/15/2018 20:23:31
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once
namespace mltools {
  /**
   * @brief Setup environment
   */
  class Env {
  public:
    Env() {}
    ~Env() {}
    
    void init(char *argv0);
  private:
    void initGlog(char *argv0);
  };
}
