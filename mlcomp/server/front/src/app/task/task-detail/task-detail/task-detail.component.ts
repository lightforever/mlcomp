import {Component} from '@angular/core';
import {TasksComponent} from "../../tasks/tasks.component";
import {Paginator} from "../../../paginator";
import {Task} from "../../../models";

@Component({
    selector: 'app-task-detail',
    templateUrl: './task-detail.component.html',
    styleUrls: ['../../tasks/tasks.component.css']
})
export class TaskDetailComponent extends TasksComponent {
    child_paginator: Paginator<Task>;

    get_filter(): any {
        let res = super.get_filter();
        res.id = this.id;
        res.type = ['User', 'Train', 'Service'];
        return res;
    }

    get id() {
        return parseInt(this.route.snapshot.paramMap.get('id'));
    }

    get filter_params_get() {
        let self = this;

        function filter_params_get_int() {
            return {
                parent: self.id,
                type: ['Service']
            }
        }

        return filter_params_get_int;

    }

    ngOnInit() {
        super.ngOnInit();

        this.child_paginator = new Paginator<Task>(
            this.service,
            this.location,
            this.filter_params_get,
            'paginator',
            true,
            false
        );

        this.child_paginator.ngOnInit();
        this.child_paginator.change.emit();
    }

    ngOnDestroy() {
        super.ngOnDestroy();
        this.child_paginator.ngOnDestroy();
    }

    onActivate(component) {
        component.task = this.id;
    }

}
