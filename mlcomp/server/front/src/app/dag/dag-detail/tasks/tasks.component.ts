import {Component, OnInit} from '@angular/core';
import {ActivatedRoute} from "@angular/router";

@Component({
    selector: 'app-tasks-host',
    templateUrl: './tasks.component.html',
    styleUrls: ['./tasks.component.css']
})
export class TasksHostComponent implements OnInit {
    private dag: number;

    constructor(private route: ActivatedRoute
    ) {
    }

    ngOnInit() {
      this.dag = parseInt(this.route.snapshot.paramMap.get('id'));
    }

}
