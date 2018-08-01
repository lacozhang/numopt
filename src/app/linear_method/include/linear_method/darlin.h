/*
 * =====================================================================================
 *
 *       Filename:  darlin.h
 *
 *    Description:  block coordinate descent algorithm
 *
 *        Version:  1.0
 *        Created:  07/29/2018 10:01:49
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  lacozhang (), lacozhang@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once

#include "system/sysutil.h"
#include "util/stringop.h"
#include "learner/bcd.h"
#include "filter/sparse_filter.h"
#include "proto/linear.pb.h"
#include "util/bitmap.h"

namespace mltools {
  DECLARE_int32(num_workers);
  namespace linear {
    typedef double Real;
    class DarlinScheduler : public BCDScheduler {
    public:
      DarlinScheduler(const Config &conf) : BCDScheduler(conf.darlin()), conf_(conf) {
        dataAssigner_.set(conf_.training_data(), FLAGS_num_workers, bcdConf_.load_local_data());
      }
      
      virtual ~DarlinScheduler() { }
      
      virtual void run() override {
        CHECK_EQ(conf_.loss().type(), LossConfig::LOGIT);
        CHECK_EQ(conf_.penalty().type(), PenaltyConfig::L1);
        LOG(INFO) << "Train l1 logistic regression";
        
        // load data
        BCDScheduler::run();
        
        auto darlin = conf_.darlin();
        int tau = darlin.max_block_delay();
        LOG(WARNING) << "max delay " << tau;
        bool randomBlkOrder = darlin.random_feature_block_order();
        if(!randomBlkOrder) {
          LOG(WARNING) << "randomized will accelerate the convergence";
        }
        
        kktFilterThreshold_ = 1e20;
        bool resetKKTFilter = false;
        int maxIter = darlin.max_pass_of_data();
        int time = exec_.time();
        int firstTime = time;
        int modelTime = featGroup_.size() * 6;
        for(int iter = 0; iter < maxIter; ++iter) {
          // select the block update order
          auto order = blkOrder_;
          if(randomBlkOrder) {
            std::random_shuffle(order.begin(), order.end());
          }
          if(iter == 0) {
            order.insert(order.begin(), priorBlkOrder_.begin(), priorBlkOrder_.end());
          }
          
          // go over all the iterators
          for (int i = 0; i < order.size(); ++i) {
            Task update;
            update.set_more(true);
            auto cmd = update.mutable_bcd();
            cmd->set_cmd(BCDCall::UPDATE_MODEL);
            
            // set bcd parameter time
            cmd->set_time(modelTime);
            modelTime += 3;
            
            // KKT filter
            if(iter == 0) {
              cmd->set_kkt_filter_threshold(kktFilterThreshold_);
              if(resetKKTFilter) {
                cmd->set_reset_kkt_filter(true);
              }
            }
            
            // block info
            auto blk = featBlk_[order[i]];
            blk.second.to(cmd->mutable_key());
            cmd->add_fea_grp(blk.first);
            
            //set command time
            update.set_time(time+1);
            if(iter == 0 && i < priorBlkOrder_.size()) {
              addWaitTime(0, firstTime, &update);
              firstTime = time;
            } else {
              addWaitTime(tau, firstTime, &update);
            }
            
            time = submit(update, kCompGroup);
          }
          
          Task eval;
          eval.set_more(false);
          eval.mutable_bcd()->set_cmd(BCDCall::EVALUATE_PROGRESS);
          eval.mutable_bcd()->set_iter(iter);
          addWaitTime(tau, firstTime, &eval);
          time = submit(eval, kCompGroup);
          wait(time);
          showProgress(iter);
          
          int k = bcdConf_.save_model_every_n_iter();
          if (k > 0 && ((iter+1) % k == 0) && conf_.has_model_output()) {
            time = saveModel(ithFile(conf_.model_output(), 0, "_it_" + std::to_string(iter)));
          }
          
          Real vio = globalProgress_[iter].violation();
          Real ratio = bcdConf_.GetExtension(kkt_filter_threshold_ratio);
          
          kktFilterThreshold_ = vio / (Real)globalTrainInfo_.num_ex() * ratio;
          
          Real rel = globalProgress_[iter].relative_obj();
          if( rel > 0 && rel <= darlin.epsilon()) {
            if(resetKKTFilter) {
              break;
            } else {
              resetKKTFilter = true;
            }
          } else {
            resetKKTFilter = false;
          }
        }
        
        for(int t=firstTime; t < time; ++t) {
          wait(t);
        }
        
        if(conf_.has_model_output()) {
          wait(saveModel(conf_.model_output()));
        }
      }

    protected:
      
      std::string showKKTFilter(int iter) {
        char buf[500];
        if (iter == -3) {
          snprintf(buf, 500, "|      KKT filter     ");
        } else if (iter == -2) {
          snprintf(buf, 500, "| threshold  #activet ");
        } else if (iter == -1) {
          snprintf(buf, 500, "+---------------------");
        } else {
          snprintf(buf, 500, "| %.1e %11llu ",
                   kktFilterThreshold_, (uint64)globalProgress_[iter].nnz_active_set());
        }
        return string(buf);
      }
      
      void showProgress(int iter) {
        int s = iter == 0 ? -3 : iter;
        for (int i = s; i <= iter; ++i) {
          string str = ShowObjective(i) + ShowKKTFilter(i) + ShowTime(i);
          NOTICE("%s", str.c_str());
        }
      }
      
      void addWaitTime(int tau, int first, Task *task) {
        int cur = task->time();
        for(int t = std::max(first, cur - 2*tau -1);
            t < std::max(first+1, cur-tau);
            ++t) {
          task->add_wait_time(t);
        }
      }
      
      Real kktFilterThreshold_;
      Config conf_;
    };
  }
}
