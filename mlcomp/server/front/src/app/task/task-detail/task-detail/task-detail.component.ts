import {Component, OnInit} from '@angular/core';
import {ActivatedRoute} from "@angular/router";

@Component({
    selector: 'app-task-detail',
    templateUrl: './task-detail.component.html',
    styleUrls: ['./task-detail.component.css']
})
export class TaskDetailComponent {
    constructor(private route: ActivatedRoute) {

    }

    onActivate(component) {
        component.task = parseInt(this.route.snapshot.paramMap.get('id'));
    }

}
