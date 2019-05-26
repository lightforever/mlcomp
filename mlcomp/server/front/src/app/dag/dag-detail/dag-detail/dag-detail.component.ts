import {AfterViewInit, Component, OnInit, ViewChild} from '@angular/core';
import {ActivatedRoute} from "@angular/router";

@Component({
  selector: 'app-dag-detail',
  templateUrl: './dag-detail.component.html',
  styleUrls: ['./dag-detail.component.css']
})
export class DagDetailComponent{
  constructor(private route: ActivatedRoute){

  }

  onActivate(component) {
    component.dag = parseInt(this.route.snapshot.paramMap.get('id'));
  }
}
