      integer max_particles 
      parameter (max_particles=12) 
      integer max_branches
      parameter (max_branches=max_particles-1)
      integer max_configs
      parameter (max_configs=10)

c     channel position
      integer config_pos,perm_pos
      common /to_config/config_pos,perm_pos

      integer max_channel
      parameter (max_channel=1000) 
