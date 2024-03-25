best_args = {
    'fl_digits': {

        
        'fedavg': {
                'local_lr': 0.001,
                'local_batch_size': 64,
        },
        
        'fedavgheal': {
                'local_lr': 0.001,
                'local_batch_size': 64,
        },
         
    },
    'fl_officecaltech': {

        'fedavg': {
            'local_lr': 0.001,
            'local_batch_size': 16,
        },
       
        'fedavgheal': {
                'local_lr': 0.001,
                'local_batch_size': 16,
        },
     
    }   
} 